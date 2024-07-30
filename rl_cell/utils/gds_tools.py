import json
import os
import subprocess
import xml.etree.ElementTree as ET

import gdsfactory as gf
from gdsfactory.component import Component


def load_gds_file(gds_file: str) -> Component:
    """
    Load a GDS file as a Component.

    Parameters:
    - gds_file: str, path to the GDS file

    Returns:
    - Component: the loaded GDS file as a Component
    """
    return gf.import_gds(gds_file)


def run_drc_on_gds_file(gds_file: str, drc_rules_file: str) -> str:
    """
    Run DRC on a GDS file using KLayout and the specified DRC rules file.

    Parameters:
    - gds_file: str, path to the GDS file
    - drc_rules_file: str, path to the DRC rules file

    Returns:
    - str: path to the error report
    """

    # for gds_file in gds_files:
    folder = os.path.dirname(gds_file)
    report_file = "error_report.txt"
    command = [
        "/Applications/klayout.app/Contents/MacOS/klayout",
        "-b",
        "-rd",
        f"input={gds_file}",
        "-rd",
        f"report={report_file}",
        "-r",
        drc_rules_file,
    ]
    subprocess.run(command)

    return os.path.join(folder, report_file)


def save_gds_file(component: Component, output_gds_filepath: str) -> None:
    """
    Save a Component as a GDS file.

    Parameters:
    - component: Component, the Component to save
    - output_gds_filepath: str, path to save the GDS file
    """
    return component.write_gds(output_gds_filepath)


def get_error_edge_pairs(drc_report_filepath: str) -> str:
    """
    Get the error edge pairs from the DRC report XML file.

    Parameters:
    - drc_report_filepath: str, path to the DRC report XML file

    Returns:
    - str: path to the output JSON file
    """

    # load XML data
    with open(drc_report_filepath) as f:
        xml_data = f.read()

    # Parse XML
    root = ET.fromstring(xml_data)

    # List to store results
    results = []

    # Find the <items> tag within <report-database>
    items = root.find("items")

    # Iterate over each item in <items>
    for item in items.findall("item"):
        category = item.find("category").text.strip(
            "'"
        )  # Remove single quotes from the category
        value = item.find(".//values/value").text

        # Split the value to get edge-pair text
        edge_pair_text = value.split(":")[1].strip()

        # Ignore OFFGRID errors
        # NOTE: Ignore the non-li errors for now
        if category.endswith("OFFGRID") or not category.startswith("li"):
            continue

        # Process edge-pair text to extract coordinate pairs
        edge_pairs = []
        for pair in edge_pair_text.split("|"):
            coords = pair.strip("()").split(";")
            edge_pair = [list(map(float, coord.split(","))) for coord in coords]
            edge_pairs.append(edge_pair)

        results.append({"category": category, "edge_pairs": edge_pairs})

    # Print results
    for result in results:
        print(result)

    output_json_filepath = os.path.join(
        os.path.dirname(drc_report_filepath), "error_edge_pairs.json"
    )

    # Save results as JSON
    with open(output_json_filepath, "w") as f:
        json.dump(results, f, indent=4)

    return output_json_filepath
