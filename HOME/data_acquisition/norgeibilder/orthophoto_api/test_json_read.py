# %%
import json
import os


def test_json_read():
    # Get the directory of the current script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the JSON file
    json_file_path = os.path.join(script_dir, "geonorge_login.json")

    # Open the JSON file
    with open(json_file_path, "r") as file:
        # Load the JSON data
        login = json.load(file)

    export_payload = {"Username": login["Username"], "Password": login["Password"]}

    print(export_payload)
