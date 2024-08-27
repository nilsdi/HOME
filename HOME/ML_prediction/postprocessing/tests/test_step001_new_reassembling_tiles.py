import unittest
from HOME.ML_prediction.postprocessing.step001_new_reassembling_tiles import (
    extract_tile_numbers,
)


class TestExtractTileNumbers(unittest.TestCase):

    def test_extract_row_col(self):
        # Example test for a basic case
        tif_name = "trondheim_kommune_2020_b_1647_45821.tif"
        row = 1647
        col = 45821
        self.assertEqual(extract_tile_numbers(tif_name), [row, row, col])

    """
    def test_edge_case(self):
        # Example test for an edge case
        input_data = "edge case input"
        expected_output = "expected result for edge case"
        self.assertEqual(extract_tile_numbers(input_data), expected_output)
    """
    # Add more test methods as needed


if __name__ == "__main__":
    unittest.main()
