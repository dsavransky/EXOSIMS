import EXOSIMS.util.vprint as vp
import unittest
from unittest.mock import patch, call


class TestVprint(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @patch("builtins.print")
    def test_vprint(self, mocked_print):
        """Test vprint

        Test method: use python's mock library to test mock output for different
        arguments
        """

        should_print = vp.vprint(True)
        dont_print = vp.vprint(False)

        should_print("AAAAAAAA")
        dont_print("BBBBBBBB")

        should_print("Savransky")
        dont_print("Savransky!")

        self.assertEqual(mocked_print.mock_calls, [call("AAAAAAAA"), call("Savransky")])


if __name__ == "__main__":
    unittest.main()
