# GUI (Planned)

Interactive frontend for the CVRowDetection pipeline.

## Planned Features

- Draw vineyard block boundaries (AOIs) on a map
- Trigger row detection pipeline on selected blocks
- Review detection results as overlays on aerial imagery
- Adjust/correct detected rows interactively
- Export results as GeoJSON

## Architecture Notes

The GUI will consume the `vinerow` package as a library. Block data flows
through the `vinerow.loaders.BlockLoader` protocol — the GUI will implement
its own loader that passes user-drawn polygons directly to the pipeline.

`map_annotator.py` in the project root is a prototype of this concept
(Leaflet-based web UI served from Python).
