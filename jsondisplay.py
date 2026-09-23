import json
import argparse
import fnmatch

import numpy as np
import plotly.graph_objects as go


# =========================================================
# Valid NeXus component classes
# =========================================================

VALID_COMPONENTS = {
    "NXsource",
    "NXsample",
    "NXdetector",
    "NXmonitor",
    "NXdisk_chopper",
    "NXguide",
    "NXslit",
    "NXaperture",
    "NXcollimator",
    "NXmoderator",
}


# =========================================================
# Colors
# =========================================================

COMPONENT_COLORS = {
    "NXsource": "red",
    "NXsample": "black",
    "NXdetector": "blue",
    "NXmonitor": "green",
    "NXdisk_chopper": "orange",
    "NXguide": "cyan",
    "NXslit": "purple",
    "NXaperture": "magenta",
    "NXcollimator": "brown",
    "NXmoderator": "yellow",
}


# =========================================================
# Utilities
# =========================================================

def attrs_to_dict(attrs):
    result = {}

    if isinstance(attrs, list):
        for a in attrs:
            if not isinstance(a, dict):
                continue

            name = a.get("name")
            value = a.get("values")

            if name is not None:
                result[name] = value

    return result


def normalize(v):
    v = np.asarray(v).astype(float)

    n = np.linalg.norm(v)

    if n == 0:
        return v

    return v / n


def point_in_limits(x, y, z, limits):
    """
    Return True if a point lies within the specified limits.

    None means that particular limit is not active.
    """

    if limits["xmin"] is not None and x < limits["xmin"]:
        return False

    if limits["xmax"] is not None and x > limits["xmax"]:
        return False

    if limits["ymin"] is not None and y < limits["ymin"]:
        return False

    if limits["ymax"] is not None and y > limits["ymax"]:
        return False

    if limits["zmin"] is not None and z < limits["zmin"]:
        return False

    if limits["zmax"] is not None and z > limits["zmax"]:
        return False

    return True


def point_mask(points, limits):
    """
    Return a boolean mask selecting points within the limits.

    points must be an Nx3 numpy array.
    """

    mask = np.ones(len(points), dtype=bool)

    if limits["xmin"] is not None:
        mask &= points[:, 0] >= limits["xmin"]

    if limits["xmax"] is not None:
        mask &= points[:, 0] <= limits["xmax"]

    if limits["ymin"] is not None:
        mask &= points[:, 1] >= limits["ymin"]

    if limits["ymax"] is not None:
        mask &= points[:, 1] <= limits["ymax"]

    if limits["zmin"] is not None:
        mask &= points[:, 2] >= limits["zmin"]

    if limits["zmax"] is not None:
        mask &= points[:, 2] <= limits["zmax"]

    return mask


# =========================================================
# Rotation matrix
# =========================================================

def rotation_matrix(axis, angle_deg):
    angle = np.deg2rad(angle_deg)

    axis = normalize(axis)

    x, y, z = axis

    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    return np.array([
        [
            x * x * C + c,
            x * y * C - z * s,
            x * z * C + y * s
        ],
        [
            y * x * C + z * s,
            y * y * C + c,
            y * z * C - x * s
        ],
        [
            z * x * C - y * s,
            z * y * C + x * s,
            z * z * C + c
        ]
    ])


# =========================================================
# Homogeneous transforms
# =========================================================

def translation_matrix(vector, value):
    T = np.eye(4)

    T[:3, 3] = value * np.asarray(vector)

    return T


def homogeneous_rotation(axis, angle_deg):
    H = np.eye(4)

    H[:3, :3] = rotation_matrix(
        axis,
        angle_deg
    )

    return H


# =========================================================
# Collect NeXus components
# =========================================================

def collect_components(
    node,
    path="",
    components=None
):
    if components is None:
        components = []

    if not isinstance(node, dict):
        return components

    name = node.get("name")

    attrs = attrs_to_dict(
        node.get("attributes", [])
    )

    nx_class = attrs.get("NX_class")

    current_path = f"{path}/{name}"

    # -----------------------------------------------------
    # Valid instrument component
    # -----------------------------------------------------

    if nx_class in VALID_COMPONENTS:
        components.append({
            "name": name,
            "node": node,
            "path": current_path,
            "nx_class": nx_class
        })

    # -----------------------------------------------------
    # Recurse
    # -----------------------------------------------------

    children = node.get("children", [])

    for child in children:
        collect_components(
            child,
            current_path,
            components
        )

    return components


# =========================================================
# Collect transformations
# =========================================================

def collect_transformations(component):
    node = component["node"]

    result = {}

    children = node.get("children", [])

    for child in children:

        if (
            child.get("type") == "group"
            and child.get("name") == "transformations"
        ):
            transforms = child.get("children", [])

            for t in transforms:

                if t.get("module") != "dataset":
                    continue

                config = t.get("config", {})

                name = config.get("name")

                attrs = attrs_to_dict(
                    t.get("attributes", [])
                )

                result[name] = {
                    "value": config.get("values"),

                    "type": attrs.get(
                        "transformation_type"
                    ),

                    "vector": attrs.get("vector"),

                    "units": attrs.get("units"),

                    "depends_on": attrs.get("depends_on")
                }

    return result


# =========================================================
# Transform -> matrix
# =========================================================

def transform_to_matrix(t):
    ttype = t["type"]

    vector = np.asarray(
        t["vector"],
        dtype=float
    )

    value = float(t["value"])

    units = t.get("units")

    # -----------------------------------------------------
    # Translation
    # -----------------------------------------------------

    if ttype == "translation":

        return translation_matrix(
            vector,
            value
        )

    # -----------------------------------------------------
    # Rotation
    # -----------------------------------------------------

    elif ttype == "rotation":

        if units in ["radians", "rad"]:
            value = np.rad2deg(value)

        return homogeneous_rotation(
            vector,
            value
        )

    return np.eye(4)


# =========================================================
# Recursive depends_on chain resolution
# =========================================================

def resolve_transform_chain(
    transforms,
    current_name,
    visited=None
):
    if visited is None:
        visited = set()

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------

    if current_name in [None, ".", ""]:
        return np.eye(4)

    # -----------------------------------------------------
    # Prevent loops
    # -----------------------------------------------------

    if current_name in visited:
        print(
            f"WARNING: transformation dependency loop "
            f"detected at '{current_name}'."
        )

        return np.eye(4)

    visited.add(current_name)

    # -----------------------------------------------------
    # Missing transform
    # -----------------------------------------------------

    if current_name not in transforms:

        print(
            f"WARNING: transform "
            f"'{current_name}' "
            f"not found locally."
        )

        return np.eye(4)

    current = transforms[current_name]

    depends_on = current.get("depends_on")

    parent_name = None

    if depends_on:
        parent_name = depends_on.split("/")[-1]

    # -----------------------------------------------------
    # Recurse upward
    # -----------------------------------------------------

    M_parent = resolve_transform_chain(
        transforms,
        parent_name,
        visited
    )

    # -----------------------------------------------------
    # Current transform
    # -----------------------------------------------------

    M_current = transform_to_matrix(current)

    # -----------------------------------------------------
    # Correct NeXus transform ordering
    # -----------------------------------------------------

    return M_parent @ M_current


# =========================================================
# Build component matrix
# =========================================================

def build_component_matrix(component):
    node = component["node"]

    transforms = collect_transformations(
        component
    )

    if not transforms:
        return np.eye(4)

    # -----------------------------------------------------
    # Explicit component depends_on
    # -----------------------------------------------------

    children = node.get("children", [])

    depends_on = None

    for child in children:

        if child.get("module") != "dataset":
            continue

        config = child.get("config", {})

        if config.get("name") == "depends_on":
            depends_on = config.get("values")
            break

    # -----------------------------------------------------
    # Infer terminal transform if missing
    # -----------------------------------------------------

    if depends_on is None:

        referenced = set()

        for t in transforms.values():

            dep = t.get("depends_on")

            if dep:
                referenced.add(
                    dep.split("/")[-1]
                )

        roots = [
            k
            for k in transforms.keys()
            if k not in referenced
        ]

        if not roots:
            return np.eye(4)

        root_name = roots[0]

    else:
        root_name = depends_on.split("/")[-1]

    print(
        f"{component['name']} "
        f"root transform -> {root_name}"
    )

    return resolve_transform_chain(
        transforms,
        root_name
    )


# =========================================================
# Child geometry extraction
# =========================================================

def extract_child_points(
    component,
    limits=None
):
    node = component["node"]

    M = build_component_matrix(component)

    point_sets = []

    children = node.get("children", [])

    datasets = {}

    # -----------------------------------------------------
    # Collect datasets
    # -----------------------------------------------------

    for child in children:

        if child.get("module") != "dataset":
            continue

        config = child.get("config", {})

        datasets[
            config.get("name")
        ] = config.get("values")

    # -----------------------------------------------------
    # XYZ offset datasets
    # -----------------------------------------------------

    xyz_candidates = [
        (
            "x_pixel_offset",
            "y_pixel_offset",
            "z_pixel_offset"
        ),
        (
            "x_offset",
            "y_offset",
            "z_offset"
        )
    ]

    for xname, yname, zname in xyz_candidates:

        if not all(
            k in datasets
            for k in [xname, yname, zname]
        ):
            continue

        x = np.asarray(
            datasets[xname],
            dtype=float
        ).flatten()

        y = np.asarray(
            datasets[yname],
            dtype=float
        ).flatten()

        z = np.asarray(
            datasets[zname],
            dtype=float
        ).flatten()

        # -------------------------------------------------
        # Construct local points
        # -------------------------------------------------

        pts = np.vstack([
            x,
            y,
            z
        ]).T

        # -------------------------------------------------
        # Homogeneous coordinates
        # -------------------------------------------------

        pts_h = np.hstack([
            pts,
            np.ones(
                (pts.shape[0], 1)
            )
        ])

        # -------------------------------------------------
        # Apply component transformation
        # -------------------------------------------------

        transformed = (
            M @ pts_h.T
        ).T[:, :3]

        # -------------------------------------------------
        # Apply spatial limits
        # -------------------------------------------------

        if limits is not None:

            mask = point_mask(
                transformed,
                limits
            )

            transformed = transformed[mask]

        # -------------------------------------------------
        # Store point set
        # -------------------------------------------------

        point_sets.append({
            "name":
                f"{component['name']} children",

            "x":
                transformed[:, 0],

            "y":
                transformed[:, 1],

            "z":
                transformed[:, 2]
        })

    return point_sets


# =========================================================
# Main plotting
# =========================================================

def plot_instrument(
    json_file,
    exclude=None,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    zmin=None,
    zmax=None
):

    if exclude is None:
        exclude = []

    # -----------------------------------------------------
    # Spatial limits
    # -----------------------------------------------------

    limits = {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "zmin": zmin,
        "zmax": zmax
    }

    # -----------------------------------------------------
    # Read JSON
    # -----------------------------------------------------

    with open(json_file, "r") as f:
        data = json.load(f)

    # -----------------------------------------------------
    # Collect components
    # -----------------------------------------------------

    components = collect_components(data)

    print("\nDetected components:\n")

    for i, c in enumerate(
        components,
        start=1
    ):
        print(
            f"{i:02d}. "
            f"{c['name']} "
            f"[{c['nx_class']}]"
        )

    # -----------------------------------------------------
    # Print active limits
    # -----------------------------------------------------

    active_limits = {
        key: value
        for key, value in limits.items()
        if value is not None
    }

    if active_limits:
        print("\nActive spatial limits:")

        for key, value in active_limits.items():
            print(
                f"  {key} = {value}"
            )

    # -----------------------------------------------------
    # Plotly figure
    # -----------------------------------------------------

    fig = go.Figure()

    # =====================================================
    # Plot components
    # =====================================================

    for c in components:

        # -------------------------------------------------
        # Exclude components using wildcard patterns
        # -------------------------------------------------

        if any(
            fnmatch.fnmatchcase(
                c["name"],
                pattern
            )
            for pattern in exclude
        ):
            print(
                f"Excluded component: "
                f"{c['name']}"
            )

            continue

        # -------------------------------------------------
        # Component transformation
        # -------------------------------------------------

        M = build_component_matrix(c)

        origin = M @ np.array([
            0,
            0,
            0,
            1
        ])

        x0, y0, z0 = origin[:3]

        # -------------------------------------------------
        # Apply spatial limits to component origin
        # -------------------------------------------------

        if not point_in_limits(
            x0,
            y0,
            z0,
            limits
        ):
            print(
                f"Component outside limits: "
                f"{c['name']} "
                f"at "
                f"x={x0:.3f}, "
                f"y={y0:.3f}, "
                f"z={z0:.3f}"
            )

            continue

        print(
            f"{c['name']} at "
            f"x={x0:.3f}, "
            f"y={y0:.3f}, "
            f"z={z0:.3f}"
        )

        # -------------------------------------------------
        # Reference point
        # -------------------------------------------------

        fig.add_trace(
            go.Scatter3d(
                x=[x0],
                y=[y0],
                z=[z0],

                mode="markers+text",

                name=c["name"],

                text=[c["name"]],

                textposition="top center",

                marker=dict(
                    size=10,

                    color=COMPONENT_COLORS.get(
                        c["nx_class"],
                        "gray"
                    ),

                    line=dict(
                        color="black",
                        width=1
                    )
                ),

                hovertext=[
                    (
                        f"{c['name']}<br>"
                        f"{c['nx_class']}<br>"
                        f"({x0:.2f}, "
                        f"{y0:.2f}, "
                        f"{z0:.2f})"
                    )
                ],

                hoverinfo="text"
            )
        )

        # -------------------------------------------------
        # Child geometry
        # -------------------------------------------------

        child_sets = extract_child_points(
            c,
            limits=limits
        )

        for child in child_sets:

            # Skip empty point sets
            if len(child["x"]) == 0:
                print(
                    f"No points within limits "
                    f"for {child['name']}"
                )

                continue

            fig.add_trace(
                go.Scatter3d(
                    x=child["x"],
                    y=child["y"],
                    z=child["z"],

                    mode="markers",

                    name=child["name"],

                    marker=dict(
                        size=1.5,

                        color=COMPONENT_COLORS.get(
                            c["nx_class"],
                            "gray"
                        ),

                        opacity=0.25
                    ),

                    hoverinfo="skip"
                )
            )

            print(
                f"Plotted "
                f"{len(child['x'])} points "
                f"for {child['name']}"
            )

    # =====================================================
    # Layout
    # =====================================================

    fig.update_layout(
        title=(
            f"NeXus JSON Geometry of "
            f"{json_file}"
        ),

        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",

            aspectmode="data"
        ),

        height=950
    )

    # =====================================================
    # Show
    # =====================================================

    fig.show()


# =========================================================
# CLI
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Display NeXus JSON geometry"
    )

    # -----------------------------------------------------
    # JSON filename
    # -----------------------------------------------------

    parser.add_argument(
        "filename",
        help="Path to JSON file"
    )

    # -----------------------------------------------------
    # Exclude components
    # -----------------------------------------------------

    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help=(
            "Component names or wildcard patterns "
            "to exclude. For example: "
            "'monitor*' excludes monitor1, monitor2, etc."
        )
    )

    # -----------------------------------------------------
    # X limits
    # -----------------------------------------------------

    parser.add_argument(
        "--xmin",
        type=float,
        default=None,
        help="Minimum X coordinate"
    )

    parser.add_argument(
        "--xmax",
        type=float,
        default=None,
        help="Maximum X coordinate"
    )

    # -----------------------------------------------------
    # Y limits
    # -----------------------------------------------------

    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Minimum Y coordinate"
    )

    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Maximum Y coordinate"
    )

    # -----------------------------------------------------
    # Z limits
    # -----------------------------------------------------

    parser.add_argument(
        "--zmin",
        type=float,
        default=None,
        help="Minimum Z coordinate"
    )

    parser.add_argument(
        "--zmax",
        type=float,
        default=None,
        help="Maximum Z coordinate"
    )

    # -----------------------------------------------------
    # Parse arguments
    # -----------------------------------------------------

    args = parser.parse_args()

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    plot_instrument(
        args.filename,

        exclude=args.exclude,

        xmin=args.xmin,
        xmax=args.xmax,

        ymin=args.ymin,
        ymax=args.ymax,

        zmin=args.zmin,
        zmax=args.zmax
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()