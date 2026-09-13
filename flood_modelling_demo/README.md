# Flood Modelling Dataset Preparation Demo

A Streamlit prototype for routine flood-risk GIS preprocessing without requiring
the user to work through a full QGIS interface.

## Project structure

```text
flood_modelling_demo/
├── streamlit_app.py
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
└── demo_data/
    ├── raster/
    │   ├── README.txt
    │   └── demo_dem.tif              # copy your demo DEM here
    └── boundary/
        ├── README.txt
        ├── demo_boundary.shp          # copy all shapefile parts here
        ├── demo_boundary.shx
        ├── demo_boundary.dbf
        └── demo_boundary.prj
```

The exact filenames are optional. If the preferred files do not exist, the app
uses the first supported raster and boundary it finds in the corresponding folders.

## Local run

```bat
conda activate prasad
cd path\to\flood_modelling_demo
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Demo workflow

At startup choose **Built-in demo**. The app loads the DEM and boundary stored
under `demo_data/`, after which the same CRS, reprojection, clipping, map preview
and download workflow is available.

Choose **Upload my own DEM** to use the normal file-upload workflow instead.

## Public deployment note

Only place data in this repository that you are allowed to redistribute. For a
public Streamlit demo, use a small public or sanitized DEM and site boundary, not
confidential client datasets.

## Streamlit Community Cloud

Push this complete folder to a GitHub repository. In Streamlit Community Cloud,
select the repository, the `main` branch, and `streamlit_app.py` as the app file.
