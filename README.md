# NeXus JSON Geometry Viewer

A Python-based 3D geometry viewer for NeXus instrument descriptions stored in JSON format.

The program reads NeXus components and their transformation chains, calculates their positions, and displays the resulting instrument geometry interactively using [Plotly](https://plotly.com/python/).

It also supports:

* Excluding components by name or wildcard pattern
* Filtering geometry by X, Y, and Z coordinate limits
* Visualizing detector/pixel geometry from offset datasets
* Recursive resolution of NeXus `depends_on` transformation chains
* Interactive 3D visualization

## Requirements

Python 3.9 or newer is recommended.

Install the required packages with:

```bash
pip install numpy plotly
```
The code can then be executed as python script.

Alternatively, by running 

```bash
pip install -e .
```

it is possible to have the functionality directely available in the terminal as

```bash
jsondisplay instrument.json
```

## Usage

Basic usage:

```bash
python jsondisplay.py instrument.json
```

where `instrument.json` is the NeXus geometry represented in JSON format.

The program will:

1. Read the JSON file.
2. Identify supported NeXus instrument components.
3. Resolve their transformation chains.
4. Calculate their global positions.
5. Plot the components in an interactive 3D Plotly figure.

## Supported NeXus Components

The following NeXus component classes are recognized:

```text
NXsource
NXsample
NXdetector
NXmonitor
NXdisk_chopper
NXguide
NXslit
NXaperture
NXcollimator
NXmoderator
```

Each component type is displayed using a different color.

## Excluding Components

Components can be excluded using the `--exclude` option.

For example:

```bash
python jsondisplay.py instrument.json --exclude monitor1
```

Multiple components can be excluded:

```bash
python jsondisplay.py instrument.json --exclude monitor1 monitor2
```

### Wildcard Exclusion

The `--exclude` option supports shell-style wildcard patterns.

For example:

```bash
python jsondisplay.py instrument.json --exclude "monitor*"
```

This excludes every component whose name starts with `monitor`, such as:

```text
monitor1
monitor2
monitora
monitor_main
monitor_detector
```

Other wildcard patterns can also be used:

```bash
python jsondisplay.py instrument.json --exclude "detector*"
```

or:

```bash
python jsondisplay.py instrument.json --exclude "*monitor*"
```

Multiple patterns can be combined:

```bash
python jsondisplay.py instrument.json \
    --exclude "monitor*" "test_detector*" "dummy*"
```

The wildcard syntax follows Python's `fnmatch` rules.

## Spatial Filtering

The geometry can be restricted to a spatial region using coordinate limits.

The following options are available:

| Option   | Description          |
| -------- | -------------------- |
| `--xmin` | Minimum X coordinate |
| `--xmax` | Maximum X coordinate |
| `--ymin` | Minimum Y coordinate |
| `--ymax` | Maximum Y coordinate |
| `--zmin` | Minimum Z coordinate |
| `--zmax` | Maximum Z coordinate |

All limits are optional.

If a limit is not specified, that direction is unbounded.

### Z Minimum

For example:

```bash
python jsondisplay.py instrument.json --zmin -100
```

Only geometry with:

```text
z >= -100
```

will be displayed.

### Z Range

To display only geometry between `z = -100` and `z = 100`:

```bash
python jsondisplay.py instrument.json --zmin -100 --zmax 100
```

### X Range

```bash
python jsondisplay.py instrument.json --xmin -50 --xmax 50
```

### Y Range

```bash
python jsondisplay.py instrument.json --ymin -20 --ymax 20
```

### Three-Dimensional Region

The limits can be combined to define a 3D bounding box:

```bash
python jsondisplay.py instrument.json \
    --xmin -50 --xmax 50 \
    --ymin -20 --ymax 20 \
    --zmin -100 --zmax 100
```

Only points inside this region are plotted.

## Combining Exclusion and Spatial Filtering

Component exclusion and spatial filtering can be used together.

For example:

```bash
python jsondisplay.py instrument.json \
    --exclude "monitor*" \
    --zmin -100 \
    --zmax 100
```

This will:

* Exclude all components whose names start with `monitor`
* Display only geometry between `z = -100` and `z = 100`

A more restrictive example:

```bash
python jsondisplay.py instrument.json \
    --exclude "monitor*" "test*" \
    --xmin -50 --xmax 50 \
    --ymin -20 --ymax 20 \
    --zmin -100 --zmax 100
```

## Coordinate System

The spatial limits are applied **after the NeXus transformation chains have been resolved**.

This means that the limits refer to the final/global coordinates displayed by the viewer rather than the original local coordinates stored in the JSON file.

For example:

```bash
python jsondisplay.py instrument.json --zmin -100
```

means:

```text
Global Z >= -100
```

This is particularly important for components whose local coordinate systems are translated or rotated relative to the instrument coordinate system.

## Transformations

The program supports the following NeXus transformation types:

* `translation`
* `rotation`

Rotation angles specified in radians are converted to degrees before constructing the rotation matrix.

Transformation dependencies are resolved recursively using the NeXus `depends_on` relationship.

For a transformation chain such as:

```text
transform3
    |
    v
transform2
    |
    v
transform1
```

the resulting transformation matrix is constructed by combining the transformations in the appropriate dependency order.

## Detector / Pixel Geometry

For components containing pixel offset datasets, the program recognizes:

```text
x_pixel_offset
y_pixel_offset
z_pixel_offset
```

and:

```text
x_offset
y_offset
z_offset
```

The individual points are transformed using the component's transformation matrix and plotted as a 3D point cloud.

Spatial limits are applied to the transformed points, so individual pixels outside the selected region are removed while points inside the region remain visible.

## Command-Line Reference

Run:

```bash
python jsondisplay.py --help
```

to display the available options.

The general syntax is:

```text
python jsondisplay.py FILE [OPTIONS]
```

Available options:

```text
--exclude PATTERN [PATTERN ...]
--xmin VALUE
--xmax VALUE
--ymin VALUE
--ymax VALUE
--zmin VALUE
--zmax VALUE
```

### Example

```bash
python jsondisplay.py instrument.json \
    --exclude "monitor*" \
    --xmin -50 \
    --xmax 50 \
    --zmin -100 \
    --zmax 100
```

## Output

The program opens an interactive Plotly 3D visualization.

The visualization provides:

* Interactive rotation
* Zooming
* Panning
* Component labels
* Component coordinates on hover
* Color-coded NeXus component types
* Detector/pixel point clouds

The coordinate axes are labeled:

```text
X [m]
Y [m]
Z [m]
```

## Notes

If no spatial limits are specified, the complete geometry is displayed:

```bash
python jsondisplay.py instrument.json
```

If no components are excluded, all supported NeXus components are displayed.

For large detector geometries, applying spatial limits can substantially reduce the number of points rendered by Plotly and can therefore improve visualization performance.

AI has been used for the code generation.