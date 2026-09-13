import os
import io
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import rasterio
from rasterio.io import MemoryFile
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
from affine import Affine

import geopandas as gpd
from shapely.geometry import box
from pyproj import Geod

import folium
from folium.raster_layers import ImageOverlay
import streamlit.components.v1 as components
from matplotlib import colormaps


# =========================================================
# REPOSITORY / DEMO DATA PATHS
# =========================================================
# All demo paths are relative to this script, so the same repository works
# on Windows, Linux, GitHub and Streamlit Community Cloud.
APP_DIR = Path(__file__).resolve().parent
DEMO_DATA_DIR = APP_DIR / "demo_data"
DEMO_RASTER_DIR = DEMO_DATA_DIR / "raster"
DEMO_BOUNDARY_DIR = DEMO_DATA_DIR / "boundary"

# Preferred filenames. If these exact names are not present, the app will
# automatically use the first supported file it finds in each folder.
PREFERRED_DEMO_RASTER = DEMO_RASTER_DIR / "demo_dem.tif"
PREFERRED_DEMO_BOUNDARY = DEMO_BOUNDARY_DIR / "demo_boundary.shp"


# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Flood Modelling Dataset Preparation",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .workflow-card {
        border: 1px solid rgba(120,120,120,0.28);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 12px;
        background: rgba(127,127,127,0.04);
    }
    .small-note {
        opacity: 0.72;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Flood Modelling Dataset Preparation")
st.caption(
    "A guided alternative to routine QGIS preprocessing: inspect, reproject, clip, visualise and export."
)


# =========================================================
# SESSION STATE
# =========================================================
def init_state():
    defaults = {
        "raster_bytes": None,
        "raster_name": None,
        "raster_info": {},
        "original_raster_bytes": None,
        "original_raster_name": None,
        "original_raster_info": {},
        "selected_action": None,
        "vector_sig": None,
        "vector_name": None,
        "vector_gdf": None,
        "vector_loaded": False,
        "vector_aligned": False,
        "clipped_bytes": None,
        "clipped_name": None,
        "clipped_info": {},
        "last_reprojected_bytes": None,
        "last_reprojected_name": None,
        "status_message": "",
        "project_mode": None,
        "demo_boundary_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# =========================================================
# GENERAL HELPERS
# =========================================================
def pretty_crs(crs):
    if crs is None:
        return "UNKNOWN"
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg else (crs.to_string() or "UNKNOWN")


def get_raster_info(raster_bytes):
    with MemoryFile(raster_bytes) as mem:
        with mem.open() as src:
            return {
                "crs_text": pretty_crs(src.crs),
                "epsg": src.crs.to_epsg() if src.crs else None,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": src.dtypes[0],
                "res": src.res,
                "bounds": src.bounds,
                "nodata": src.nodata,
            }


def crs_match(crs_a, crs_b):
    if crs_a is None or crs_b is None:
        return False
    try:
        return CRS.from_user_input(crs_a) == CRS.from_user_input(crs_b)
    except Exception:
        return str(crs_a) == str(crs_b)


def raster_extent_area_km2(src):
    """Approximate area of the raster extent, not the count of valid-data pixels."""
    if src.crs and src.crs.is_projected:
        px_w = abs(src.transform.a)
        px_h = abs(src.transform.e)
        return (src.width * px_w * src.height * px_h) / 1e6

    geod = Geod(ellps="WGS84")
    b = src.bounds
    polygon = box(b.left, b.bottom, b.right, b.top)
    lon, lat = polygon.exterior.coords.xy
    area, _ = geod.polygon_area_perimeter(lon, lat)
    return abs(area) / 1e6


def read_kml_path(kml_path):
    """
    Read a KML file as a GeoDataFrame.

    KML is defined in geographic WGS 84 coordinates. Some GDAL/Fiona
    installations do not explicitly return the CRS, so EPSG:4326 is assigned
    only when the KML reader returns no CRS metadata.

    If the KML contains several layers/folders, readable layers are combined.
    """
    frames = []
    layer_errors = []

    # Fiona sometimes ships with KML support present but not enabled for writing.
    # Enabling read/write here also makes older GeoPandas/Fiona stacks more reliable.
    try:
        import fiona
        fiona.drvsupport.supported_drivers["KML"] = "rw"

        try:
            layers = fiona.listlayers(kml_path)
        except Exception:
            layers = []

        for layer in layers:
            try:
                g = gpd.read_file(kml_path, layer=layer)
                if not g.empty:
                    frames.append(g)
            except Exception as exc:
                layer_errors.append(str(exc))
    except Exception:
        layers = []

    # If layer enumeration did not work, try a normal GeoPandas read.
    if not frames:
        try:
            gdf = gpd.read_file(kml_path)
            if not gdf.empty:
                frames.append(gdf)
        except Exception as exc:
            message = (
                "Could not read the KML file. Your GDAL/Fiona installation may not "
                "include KML support. Install/update geopandas, fiona and gdal from "
                "conda-forge. Original error: " + str(exc)
            )
            raise ValueError(message) from exc

    if not frames:
        raise ValueError("The KML file contains no readable geographic features.")

    # KML layers should all use WGS84. Bring all readable layers into one GDF.
    base_crs = next((frame.crs for frame in frames if frame.crs is not None), None)
    if base_crs is None:
        base_crs = CRS.from_epsg(4326)

    aligned_frames = []
    for frame in frames:
        if frame.crs is None:
            frame = frame.set_crs(epsg=4326)
        elif not crs_match(frame.crs, base_crs):
            frame = frame.to_crs(base_crs)
        aligned_frames.append(frame)

    gdf = gpd.GeoDataFrame(
        pd.concat(aligned_frames, ignore_index=True),
        crs=base_crs,
    )

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    return gdf


def keep_polygon_boundaries(gdf):
    """
    Keep polygonal features suitable for raster clipping.

    KML/KMZ files can contain points, paths and polygons together. Raster clipping
    requires an area boundary, so point/line features are ignored when polygons
    are also present. If no polygons exist, a clear error is returned.
    """
    if gdf is None or gdf.empty:
        raise ValueError("Boundary layer is empty.")

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError("Boundary layer contains no valid geometry.")

    polygon_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])

    if polygon_mask.any():
        gdf = gdf.loc[polygon_mask].copy()
    else:
        raise ValueError(
            "The uploaded boundary contains no Polygon or MultiPolygon geometry. "
            "A polygon boundary is required to clip a raster."
        )

    return gdf


def read_vector_upload(uploaded_file):
    """
    Read a site boundary uploaded through Streamlit.

    Supported formats:
    - ZIP shapefile (.zip containing .shp, .shx, .dbf and preferably .prj)
    - GeoPackage (.gpkg)
    - GeoJSON (.geojson / .json)
    - KML (.kml)
    - KMZ (.kmz; KML compressed inside a ZIP container)
    """
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".zip"):
        with tempfile.TemporaryDirectory() as td:
            zip_path = os.path.join(td, "boundary.zip")
            with open(zip_path, "wb") as f:
                f.write(data)

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(td)

            shp_files = []
            for root, _, files in os.walk(td):
                for filename in files:
                    if filename.lower().endswith(".shp"):
                        shp_files.append(os.path.join(root, filename))

            if not shp_files:
                raise ValueError(
                    "The ZIP does not contain a .shp file. Include .shp, .shx, .dbf and .prj. "
                    "If your file is a KMZ, upload it with the .kmz extension rather than .zip."
                )

            return keep_polygon_boundaries(gpd.read_file(shp_files[0]))

    if name.endswith(".gpkg"):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(data)
            return keep_polygon_boundaries(gpd.read_file(path))

    if name.endswith(".geojson") or name.endswith(".json"):
        return keep_polygon_boundaries(gpd.read_file(io.BytesIO(data)))

    if name.endswith(".kml"):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "boundary.kml")
            with open(path, "wb") as f:
                f.write(data)
            return keep_polygon_boundaries(read_kml_path(path))

    if name.endswith(".kmz"):
        with tempfile.TemporaryDirectory() as td:
            kmz_path = os.path.join(td, "boundary.kmz")
            with open(kmz_path, "wb") as f:
                f.write(data)

            try:
                with zipfile.ZipFile(kmz_path, "r") as zf:
                    zf.extractall(td)
            except zipfile.BadZipFile as exc:
                raise ValueError("The uploaded KMZ is not a valid KMZ/ZIP archive.") from exc

            kml_files = []
            for root, _, files in os.walk(td):
                for filename in files:
                    if filename.lower().endswith(".kml"):
                        kml_files.append(os.path.join(root, filename))

            if not kml_files:
                raise ValueError("The KMZ archive does not contain a KML file.")

            # Google Earth commonly stores the primary document as doc.kml.
            kml_files.sort(
                key=lambda path: (
                    os.path.basename(path).lower() != "doc.kml",
                    path.lower(),
                )
            )

            return keep_polygon_boundaries(read_kml_path(kml_files[0]))

    raise ValueError(
        "Supported vector formats: ZIP shapefile, GPKG, GeoJSON, KML and KMZ."
    )


def read_vector_path(path):
    """Read a boundary file stored inside the repository/demo_data folder."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Boundary file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".shp":
        return keep_polygon_boundaries(gpd.read_file(path))

    if suffix == ".gpkg":
        return keep_polygon_boundaries(gpd.read_file(path))

    if suffix in {".geojson", ".json"}:
        return keep_polygon_boundaries(gpd.read_file(path))

    if suffix == ".kml":
        return keep_polygon_boundaries(read_kml_path(str(path)))

    if suffix == ".kmz":
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(td)

            kml_files = list(Path(td).rglob("*.kml"))
            if not kml_files:
                raise ValueError("The KMZ archive does not contain a KML file.")

            kml_files.sort(
                key=lambda p: (p.name.lower() != "doc.kml", str(p).lower())
            )
            return keep_polygon_boundaries(read_kml_path(str(kml_files[0])))

    if suffix == ".zip":
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(td)

            shp_files = list(Path(td).rglob("*.shp"))
            if not shp_files:
                raise ValueError("The ZIP does not contain a shapefile (.shp).")
            return keep_polygon_boundaries(gpd.read_file(shp_files[0]))

    raise ValueError(
        "Supported demo boundary formats: SHP, ZIP shapefile, GPKG, GeoJSON, KML and KMZ."
    )


def find_demo_raster():
    """Return the preferred demo DEM, or the first GeoTIFF in demo_data/raster."""
    if PREFERRED_DEMO_RASTER.exists():
        return PREFERRED_DEMO_RASTER

    if not DEMO_RASTER_DIR.exists():
        return None

    candidates = sorted(
        list(DEMO_RASTER_DIR.glob("*.tif"))
        + list(DEMO_RASTER_DIR.glob("*.tiff"))
    )
    return candidates[0] if candidates else None


def find_demo_boundary():
    """Return preferred demo boundary, or the first supported boundary file."""
    if PREFERRED_DEMO_BOUNDARY.exists():
        return PREFERRED_DEMO_BOUNDARY

    if not DEMO_BOUNDARY_DIR.exists():
        return None

    priority = ["*.shp", "*.geojson", "*.gpkg", "*.kml", "*.kmz", "*.zip"]
    for pattern in priority:
        matches = sorted(DEMO_BOUNDARY_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_demo_into_session():
    """Load the repository DEM and boundary into Streamlit session state."""
    raster_path = find_demo_raster()
    boundary_path = find_demo_boundary()

    if raster_path is None:
        raise FileNotFoundError(
            f"No demo GeoTIFF found in {DEMO_RASTER_DIR}. "
            "Copy a .tif/.tiff file into that folder."
        )

    if boundary_path is None:
        raise FileNotFoundError(
            f"No demo boundary found in {DEMO_BOUNDARY_DIR}. "
            "Copy a shapefile (all components) or another supported boundary into that folder."
        )

    raster_bytes = raster_path.read_bytes()
    raster_info = get_raster_info(raster_bytes)
    if raster_info["crs_text"] == "UNKNOWN":
        raise ValueError("The demo raster has no CRS metadata.")

    boundary_gdf = read_vector_path(boundary_path)
    if boundary_gdf.crs is None:
        raise ValueError(
            "The demo boundary has no CRS. For a shapefile, ensure the .prj file is present."
        )

    with MemoryFile(raster_bytes) as mem:
        with mem.open() as src:
            raster_crs = src.crs

    st.session_state.raster_bytes = raster_bytes
    st.session_state.raster_name = raster_path.name
    st.session_state.raster_info = raster_info
    st.session_state.original_raster_bytes = raster_bytes
    st.session_state.original_raster_name = raster_path.name
    st.session_state.original_raster_info = raster_info

    st.session_state.vector_sig = ("demo", str(boundary_path.resolve()))
    st.session_state.vector_name = boundary_path.name
    st.session_state.vector_gdf = boundary_gdf
    st.session_state.vector_loaded = True
    st.session_state.vector_aligned = crs_match(boundary_gdf.crs, raster_crs)
    st.session_state.demo_boundary_path = str(boundary_path)

    st.session_state.project_mode = "Built-in demo"
    st.session_state.status_message = (
        f"Demo project loaded. Raster CRS is {raster_info['crs_text']}. "
        "The demo site boundary is also loaded. Choose what you want to do next."
    )


# =========================================================
# RASTER PROCESSING
# =========================================================
def reproject_raster_bytes(raster_bytes, target_epsg):
    target_crs = CRS.from_epsg(int(target_epsg))

    with MemoryFile(raster_bytes) as src_mem:
        with src_mem.open() as src:
            if src.crs is None:
                raise ValueError("The raster has no CRS and cannot be reprojected safely.")

            transform, width, height = calculate_default_transform(
                src.crs,
                target_crs,
                src.width,
                src.height,
                *src.bounds,
            )

            total_pixels = int(width) * int(height)
            if total_pixels > 300_000_000:
                raise ValueError(
                    "The output would exceed 300 million pixels. Check the source CRS and extent."
                )

            profile = src.profile.copy()
            profile.update(
                {
                    "driver": "GTiff",
                    "crs": target_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "compress": "LZW",
                    "tiled": True,
                    "bigtiff": "IF_SAFER",
                }
            )

            out_mem = MemoryFile()
            with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS", GDAL_CACHEMAX=512):
                with out_mem.open(**profile) as dst:
                    for band in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, band),
                            destination=rasterio.band(dst, band),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            src_nodata=src.nodata,
                            dst_nodata=src.nodata,
                            resampling=Resampling.bilinear,
                            num_threads=0,
                            warp_mem_limit=512 * 1024 * 1024,
                        )

            output = out_mem.read()
            out_mem.close()
            return output


def clip_raster_bytes(raster_bytes, gdf):
    with MemoryFile(raster_bytes) as src_mem:
        with src_mem.open() as src:
            if src.crs is None:
                raise ValueError("Raster CRS is missing.")
            if gdf.crs is None:
                raise ValueError("Boundary CRS is missing.")

            if not crs_match(gdf.crs, src.crs):
                gdf = gdf.to_crs(src.crs)

            shapes = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
            if not shapes:
                raise ValueError("The boundary contains no valid geometry.")

            out_img, out_transform = mask(
                src,
                shapes=shapes,
                crop=True,
                nodata=src.nodata,
            )

            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "height": out_img.shape[1],
                    "width": out_img.shape[2],
                    "transform": out_transform,
                    "driver": "GTiff",
                    "compress": "LZW",
                    "tiled": True,
                    "bigtiff": "IF_SAFER",
                }
            )

            out_mem = MemoryFile()
            with out_mem.open(**out_meta) as dst:
                dst.write(out_img)

            output = out_mem.read()
            out_mem.close()
            return output


# =========================================================
# MAP PREVIEW HELPERS
# =========================================================
@st.cache_data(show_spinner=False)
def raster_preview_for_map(raster_bytes, max_dimension=1000):
    """
    Creates a lightweight EPSG:4326 preview for the browser map.
    The source GeoTIFF itself is not altered.
    """
    with MemoryFile(raster_bytes) as src_mem:
        with src_mem.open() as src:
            if src.crs is None:
                raise ValueError("Raster has no CRS, so it cannot be placed on a map.")

            dst_crs = CRS.from_epsg(4326)
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,
                dst_crs,
                src.width,
                src.height,
                *src.bounds,
            )

            original_width = dst_width
            original_height = dst_height

            scale = max(original_width, original_height) / float(max_dimension)
            if scale > 1:
                dst_width = max(1, int(round(original_width / scale)))
                dst_height = max(1, int(round(original_height / scale)))
                dst_transform = dst_transform * Affine.scale(
                    original_width / dst_width,
                    original_height / dst_height,
                )

            destination = np.full((dst_height, dst_width), np.nan, dtype="float32")

            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

            valid = np.isfinite(destination)
            if not np.any(valid):
                raise ValueError("No valid pixels were available for map preview.")

            values = destination[valid]
            low = float(np.nanpercentile(values, 2))
            high = float(np.nanpercentile(values, 98))
            if high <= low:
                low = float(np.nanmin(values))
                high = float(np.nanmax(values))
            if high <= low:
                high = low + 1.0

            normalised = np.clip((destination - low) / (high - low), 0, 1)
            rgba = (colormaps["terrain"](normalised) * 255).astype(np.uint8)
            rgba[~valid, 3] = 0

            west, south, east, north = array_bounds(
                dst_height,
                dst_width,
                dst_transform,
            )

            return {
                "rgba": rgba,
                "bounds": [[south, west], [north, east]],
                "center": [(south + north) / 2, (west + east) / 2],
                "low": low,
                "high": high,
            }


def add_north_arrow(map_object):
    html = """
    <div style="
        position: fixed;
        top: 14px;
        right: 14px;
        z-index: 9999;
        background: rgba(255,255,255,0.92);
        border: 1px solid #444;
        border-radius: 7px;
        padding: 5px 9px;
        text-align: center;
        font-family: Arial, sans-serif;
        font-size: 14px;
        line-height: 15px;
        color: #111;">
        <b>N</b><br><span style="font-size:22px;">▲</span>
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(html))


def build_folium_map(raster_bytes, boundary_gdf=None, layer_name="Raster"):
    preview = raster_preview_for_map(raster_bytes)

    m = folium.Map(
        location=preview["center"],
        tiles="CartoDB positron",
        control_scale=True,
        zoom_control=True,
    )

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    ImageOverlay(
        image=preview["rgba"],
        bounds=preview["bounds"],
        opacity=0.78,
        name=layer_name,
        interactive=False,
        cross_origin=False,
        zindex=2,
    ).add_to(m)

    if boundary_gdf is not None and not boundary_gdf.empty:
        if boundary_gdf.crs is None:
            raise ValueError("Boundary CRS is missing.")
        boundary_wgs = boundary_gdf.to_crs(epsg=4326)
        folium.GeoJson(
            boundary_wgs.__geo_interface__,
            name="Site boundary",
            style_function=lambda feature: {
                "color": "#d62728",
                "weight": 3,
                "fillColor": "#d62728",
                "fillOpacity": 0.05,
            },
        ).add_to(m)

    m.fit_bounds(preview["bounds"], padding=(20, 20))
    add_north_arrow(m)
    folium.LayerControl(collapsed=False).add_to(m)

    return m, preview


def show_raster_map(raster_bytes, boundary_gdf=None, layer_name="Raster", height=590):
    try:
        fmap, preview = build_folium_map(
            raster_bytes,
            boundary_gdf=boundary_gdf,
            layer_name=layer_name,
        )
        components.html(fmap.get_root().render(), height=height, scrolling=False)
        st.caption(
            f"Browser preview uses Band 1 and is stretched between the 2nd and 98th percentiles "
            f"({preview['low']:.2f} to {preview['high']:.2f}). The downloaded GeoTIFF is unchanged."
        )
    except Exception as exc:
        st.error(f"Map preview could not be created: {exc}")


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Workspace")

    if st.session_state.project_mode:
        st.write("Mode:", st.session_state.project_mode)

    if st.session_state.raster_name:
        st.write("Working raster")
        st.write(st.session_state.raster_name)
        st.write("CRS:", st.session_state.raster_info.get("crs_text", ""))
    else:
        st.write("Working raster: not loaded")

    if st.session_state.vector_name:
        st.write("Boundary")
        st.write(st.session_state.vector_name)
        st.write("Aligned:", "Yes" if st.session_state.vector_aligned else "No")

    if st.session_state.clipped_name:
        st.write("Latest clipped output")
        st.write(st.session_state.clipped_name)

    if st.session_state.last_reprojected_bytes is not None or st.session_state.clipped_bytes is not None:
        st.markdown("---")
        st.subheader("Downloads")

        if st.session_state.last_reprojected_bytes is not None:
            st.download_button(
                "Reprojected raster",
                data=st.session_state.last_reprojected_bytes,
                file_name=st.session_state.last_reprojected_name,
                mime="image/tiff",
                use_container_width=True,
                key="sidebar_download_reprojected",
            )

        if st.session_state.clipped_bytes is not None:
            st.download_button(
                "Clipped raster",
                data=st.session_state.clipped_bytes,
                file_name=st.session_state.clipped_name,
                mime="image/tiff",
                use_container_width=True,
                key="sidebar_download_clipped",
            )

    st.markdown("---")

    if st.button("Restart workflow", use_container_width=True):
        keys = list(st.session_state.keys())
        for key in keys:
            del st.session_state[key]
        st.rerun()


# =========================================================
# STEP 1 - START PROJECT
# =========================================================
if st.session_state.raster_bytes is None:
    st.markdown(
        '<div class="workflow-card"><b>Start:</b> Open the built-in demo project or upload your own DEM.</div>',
        unsafe_allow_html=True,
    )

    start_mode = st.radio(
        "How would you like to begin?",
        ["Built-in demo", "Upload my own DEM"],
        horizontal=True,
        key="start_project_mode",
    )

    if start_mode == "Built-in demo":
        demo_raster = find_demo_raster()
        demo_boundary = find_demo_boundary()

        st.write("Demo data location")
        st.code(
            "demo_data/\n"
            "  raster/      <- copy your demo DEM here\n"
            "  boundary/    <- copy your demo shapefile/KML/KMZ here",
            language=None,
        )

        c1, c2 = st.columns(2)
        c1.write(
            "Raster: " + (demo_raster.name if demo_raster else "Not found")
        )
        c2.write(
            "Boundary: " + (demo_boundary.name if demo_boundary else "Not found")
        )

        if demo_raster is None:
            st.warning(
                "No demo raster found. Copy a .tif or .tiff into demo_data/raster/. "
                "The preferred name is demo_dem.tif, but any GeoTIFF will work."
            )

        if demo_boundary is None:
            st.warning(
                "No demo boundary found. Copy the complete shapefile set into "
                "demo_data/boundary/ (or use GeoJSON, GPKG, KML, KMZ or ZIP shapefile)."
            )

        if st.button(
            "Load demo project",
            type="primary",
            use_container_width=True,
            disabled=(demo_raster is None or demo_boundary is None),
        ):
            try:
                load_demo_into_session()
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load the demo project: {exc}")

    else:
        upload = st.file_uploader(
            "Upload GeoTIFF",
            type=["tif", "tiff"],
            key="main_raster_upload",
        )

        if upload is not None:
            data = upload.getvalue()

            try:
                info = get_raster_info(data)
                if info["crs_text"] == "UNKNOWN":
                    st.error("The raster has no CRS metadata. Assign the correct CRS first.")
                    st.stop()

                st.session_state.raster_bytes = data
                st.session_state.raster_name = upload.name
                st.session_state.raster_info = info
                st.session_state.original_raster_bytes = data
                st.session_state.original_raster_name = upload.name
                st.session_state.original_raster_info = info
                st.session_state.project_mode = "User upload"
                st.session_state.status_message = (
                    f"Raster loaded. Current CRS is {info['crs_text']}. Choose what you want to do next."
                )
                st.rerun()

            except Exception as exc:
                st.error(f"Could not read the raster: {exc}")

    st.stop()


# =========================================================
# MAIN WORKSPACE
# =========================================================
left, right = st.columns([0.38, 0.62], gap="large")

with left:
    st.markdown(
        f'<div class="workflow-card"><b>Assistant:</b> {st.session_state.status_message or "What do you want to do with the raster?"}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Choose an action")

    c1, c2 = st.columns(2)
    check_crs = c1.button("A. Check CRS", use_container_width=True)
    check_area = c2.button("B. Check area / extent", use_container_width=True)

    c3, c4 = st.columns(2)
    reproject_action = c3.button("C. Reproject raster", use_container_width=True)
    clip_action = c4.button("D. Clip to site", use_container_width=True)

    typed = st.chat_input("You can also type: crs, area, reproject, clip")

    if check_crs:
        st.session_state.selected_action = "crs"
    elif check_area:
        st.session_state.selected_action = "area"
    elif reproject_action:
        st.session_state.selected_action = "reproject"
    elif clip_action:
        st.session_state.selected_action = "clip"
    elif typed:
        command = typed.strip().lower()
        if command in {"a", "crs", "check crs"}:
            st.session_state.selected_action = "crs"
        elif command in {"b", "area", "extent", "check area"}:
            st.session_state.selected_action = "area"
        elif command in {"c", "reproject", "projection", "change crs"}:
            st.session_state.selected_action = "reproject"
        elif command in {"d", "clip", "site", "clip to site"}:
            st.session_state.selected_action = "clip"
        else:
            st.warning("Try: crs, area, reproject or clip.")

    action = st.session_state.selected_action

    # -----------------------------------------------------
    # A - CHECK CRS
    # -----------------------------------------------------
    if action == "crs":
        info = st.session_state.raster_info
        st.markdown("### CRS result")
        st.success(f"Current raster CRS: {info['crs_text']}")
        st.write("Resolution:", info["res"])
        st.write("Raster size:", f"{info['width']} x {info['height']} pixels")

        if info["epsg"] in [32643, 32644, 32645]:
            st.info("The raster is already in one of the configured UTM zones.")
        else:
            st.info("If you need a projected working CRS, choose Reproject raster next.")

    # -----------------------------------------------------
    # B - AREA / EXTENT
    # -----------------------------------------------------
    if action == "area":
        with MemoryFile(st.session_state.raster_bytes) as mem:
            with mem.open() as src:
                area_km2 = raster_extent_area_km2(src)

        info = st.session_state.raster_info
        st.markdown("### Raster extent")
        st.metric("Approximate extent area", f"{area_km2:,.2f} km²")
        st.write("Bounds:", info["bounds"])
        st.write("Resolution:", info["res"])
        st.caption("Area is calculated from the raster extent, not from valid-data pixels only.")

    # -----------------------------------------------------
    # C - REPROJECT
    # -----------------------------------------------------
    if action == "reproject":
        st.markdown("### Reproject raster")
        st.write("Current CRS:", st.session_state.raster_info.get("crs_text"))

        target_label = st.radio(
            "Choose target CRS",
            [
                "EPSG:32643 - WGS 84 / UTM zone 43N",
                "EPSG:32644 - WGS 84 / UTM zone 44N",
                "EPSG:32645 - WGS 84 / UTM zone 45N",
                "Other EPSG code",
            ],
            key="target_crs_choice",
        )

        if target_label.startswith("EPSG:32643"):
            target_epsg = 32643
        elif target_label.startswith("EPSG:32644"):
            target_epsg = 32644
        elif target_label.startswith("EPSG:32645"):
            target_epsg = 32645
        else:
            target_epsg = int(
                st.number_input(
                    "Enter EPSG code",
                    min_value=1000,
                    max_value=999999,
                    value=32644,
                    step=1,
                )
            )

        if st.button("Apply reprojection", type="primary", use_container_width=True):
            try:
                with st.spinner("Reprojecting raster..."):
                    output = reproject_raster_bytes(
                        st.session_state.raster_bytes,
                        target_epsg,
                    )

                base = os.path.splitext(st.session_state.raster_name)[0]
                output_name = f"{base}_EPSG{target_epsg}.tif"

                st.session_state.raster_bytes = output
                st.session_state.raster_name = output_name
                st.session_state.raster_info = get_raster_info(output)
                st.session_state.last_reprojected_bytes = output
                st.session_state.last_reprojected_name = output_name

                # A new working raster invalidates any previous clip output.
                st.session_state.clipped_bytes = None
                st.session_state.clipped_name = None
                st.session_state.clipped_info = {}

                # Boundary is preserved but must be checked against the new CRS again.
                st.session_state.vector_aligned = False

                st.session_state.status_message = (
                    f"Reprojection complete. The working raster is now {pretty_crs(CRS.from_epsg(target_epsg))}."
                )
                st.rerun()

            except Exception as exc:
                st.error(f"Reprojection failed: {exc}")

        if st.session_state.last_reprojected_bytes is not None:
            st.download_button(
                "Download reprojected raster",
                data=st.session_state.last_reprojected_bytes,
                file_name=st.session_state.last_reprojected_name,
                mime="image/tiff",
                use_container_width=True,
            )

    # -----------------------------------------------------
    # D - CLIP
    # -----------------------------------------------------
    if action == "clip":
        st.markdown("### Clip raster to site boundary")

        if st.session_state.vector_loaded and st.session_state.vector_name:
            st.info(
                f"Boundary already loaded: {st.session_state.vector_name}. "
                "You can use it directly or upload another boundary below to replace it."
            )
            boundary_upload_label = "Optional: replace site boundary"
        else:
            boundary_upload_label = "Upload site boundary"

        vector_upload = st.file_uploader(
            boundary_upload_label,
            type=["zip", "gpkg", "geojson", "json", "kml", "kmz"],
            help=(
                "Supported: ZIP shapefile, GPKG, GeoJSON, KML and KMZ. "
                "For shapefiles, ZIP the .shp, .shx, .dbf and .prj together. "
                "KML/KMZ are treated as WGS 84 and can be reprojected to the raster CRS inside the workflow."
            ),
            key="site_boundary_upload",
        )

        if vector_upload is not None:
            vector_bytes = vector_upload.getvalue()
            vector_sig = (
                vector_upload.name,
                len(vector_bytes),
                vector_bytes[:64],
            )

            if st.session_state.vector_sig != vector_sig:
                try:
                    gdf = read_vector_upload(vector_upload)
                    if gdf.empty:
                        raise ValueError("Boundary layer is empty.")
                    if gdf.crs is None:
                        raise ValueError("Boundary CRS is missing. For shapefiles, check the .prj file. KML/KMZ should normally be WGS 84.")

                    st.session_state.vector_sig = vector_sig
                    st.session_state.vector_name = vector_upload.name
                    st.session_state.vector_gdf = gdf
                    st.session_state.vector_loaded = True
                    st.session_state.vector_aligned = False
                    st.session_state.clipped_bytes = None
                    st.session_state.clipped_name = None
                    st.session_state.clipped_info = {}
                    st.rerun()

                except Exception as exc:
                    st.error(f"Could not read the boundary: {exc}")

        if st.session_state.vector_loaded and st.session_state.vector_gdf is not None:
            gdf = st.session_state.vector_gdf
            raster_crs = None
            with MemoryFile(st.session_state.raster_bytes) as mem:
                with mem.open() as src:
                    raster_crs = src.crs

            st.write("Raster CRS:", pretty_crs(raster_crs))
            st.write("Boundary CRS:", pretty_crs(gdf.crs))

            if not crs_match(gdf.crs, raster_crs) and not st.session_state.vector_aligned:
                st.warning("CRS mismatch detected between the raster and site boundary.")
                answer = st.radio(
                    "Reproject the boundary to match the raster?",
                    ["Yes", "No"],
                    horizontal=True,
                    key="boundary_reproject_choice",
                )

                if answer == "Yes":
                    if st.button(
                        "Reproject boundary to raster CRS",
                        use_container_width=True,
                    ):
                        try:
                            st.session_state.vector_gdf = gdf.to_crs(raster_crs)
                            st.session_state.vector_aligned = True
                            st.session_state.status_message = (
                                "Boundary CRS now matches the working raster. Review the overlay on the map, then clip."
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Boundary reprojection failed: {exc}")
                else:
                    st.info("Clip is paused until both layers use the same CRS.")
            else:
                st.session_state.vector_aligned = True

            if st.session_state.vector_aligned:
                st.success("Raster and boundary CRS are aligned.")
                st.write("Review the boundary overlay on the map before clipping.")

                if st.button(
                    "Clip raster to site boundary",
                    type="primary",
                    use_container_width=True,
                ):
                    try:
                        with st.spinner("Clipping raster..."):
                            clipped = clip_raster_bytes(
                                st.session_state.raster_bytes,
                                st.session_state.vector_gdf,
                            )

                        base = os.path.splitext(st.session_state.raster_name)[0]
                        clipped_name = f"{base}_CLIPPED.tif"

                        st.session_state.clipped_bytes = clipped
                        st.session_state.clipped_name = clipped_name
                        st.session_state.clipped_info = get_raster_info(clipped)
                        st.session_state.status_message = (
                            "Clipping complete. The clipped raster is displayed on the map and is ready to download."
                        )
                        st.rerun()

                    except Exception as exc:
                        st.error(f"Clipping failed: {exc}")

        if st.session_state.clipped_bytes is not None:
            st.success("Clipped raster created successfully.")
            st.download_button(
                "Download clipped raster",
                data=st.session_state.clipped_bytes,
                file_name=st.session_state.clipped_name,
                mime="image/tiff",
                use_container_width=True,
            )


# =========================================================
# MAP CANVAS
# =========================================================
# =========================================================
# MAP CANVAS
# =========================================================
with right:
    st.subheader("Map canvas")

    boundary_for_map = None
    if st.session_state.vector_loaded and st.session_state.vector_gdf is not None:
        boundary_for_map = st.session_state.vector_gdf

    # -----------------------------------------------------
    # AFTER CLIPPING
    # Show only the clipped raster and automatically zoom to
    # its new extent. The original/working raster is not added
    # to this map, so there is no full DEM layer left switched on.
    # -----------------------------------------------------
    if st.session_state.clipped_bytes is not None:
        show_raster_map(
            st.session_state.clipped_bytes,
            boundary_gdf=None,
            layer_name="Clipped raster",
            height=640,
        )

        st.caption(
            "Clipping is complete. The map now shows only the clipped raster "
            "and is automatically fitted to the clipped extent."
        )

        st.markdown("### Before / after clipping")
        before = st.session_state.raster_info
        after = st.session_state.clipped_info
        comparison = {
            "Property": ["CRS", "Width", "Height", "Resolution"],
            "Before clip": [
                before.get("crs_text"),
                before.get("width"),
                before.get("height"),
                str(before.get("res")),
            ],
            "After clip": [
                after.get("crs_text"),
                after.get("width"),
                after.get("height"),
                str(after.get("res")),
            ],
        }
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    # -----------------------------------------------------
    # AFTER REPROJECTION, BEFORE CLIPPING
    # Keep the comparison tabs useful at this stage.
    # -----------------------------------------------------
    elif st.session_state.last_reprojected_bytes is not None:
        tab0, tab1 = st.tabs(["Original input", "Working raster"])

        with tab0:
            show_raster_map(
                st.session_state.original_raster_bytes,
                boundary_gdf=boundary_for_map,
                layer_name="Original raster",
            )

        with tab1:
            show_raster_map(
                st.session_state.raster_bytes,
                boundary_gdf=boundary_for_map,
                layer_name="Reprojected working raster",
            )

        st.caption(
            "Reprojection changes the coordinate system of the working GeoTIFF. "
            "The map preview is always rendered in WGS84 for browser display."
        )

    # -----------------------------------------------------
    # NORMAL WORKING VIEW
    # Before clipping, show the working raster and uploaded
    # site boundary so overlap can be checked visually.
    # -----------------------------------------------------
    else:
        show_raster_map(
            st.session_state.raster_bytes,
            boundary_gdf=boundary_for_map,
            layer_name="Working raster",
        )

        if boundary_for_map is not None:
            st.caption(
                "The red outline is the uploaded site boundary. It is displayed before clipping "
                "so you can visually confirm overlap."
            )
