import unittest
import os
from tests.TestSupport.Info import resource_path
import json
from EXOSIMS.util.CheckScript import CheckScript


class TestCheckScript(unittest.TestCase):
    """
    Tests the CheckScript tool.
    """

    def setUp(self):
        self.dev_null = open(os.devnull, "w")
        # use template_minimal.json
        self.script1path = resource_path("test-scripts/template_minimal.json")
        with open(self.script1path, "r") as f:
            self.script1dict = json.loads(f.read())

        self.script2path = resource_path("test-scripts/template_prototype_testing.json")
        with open(self.script2path, "r") as f:
            self.script2dict = json.loads(f.read())

        self.script3path = resource_path(
            "test-scripts/template_completeness_testing.json"
        )
        with open(self.script3path, "r") as f:
            self.script3dict = json.loads(f.read())

        self.script4path = resource_path("test-scripts/incorrectly_formatted.json")

    def tearDown(self):
        self.dev_null.close()

    def test_incorrect_format(self):
        """
        Tests that an incorrectly formatted json raises an error.
        """

        self.assertRaises(
            ValueError, lambda: CheckScript(self.script4path, self.script1dict)
        )

    def test_write_file(self):
        """
        Tests that a file has been written.
        """
        # use template_minimal.json and dictionary equivalent
        CS = CheckScript(self.script1path, self.script1dict)
        filename = "CheckScript.test"
        CS.write_file(filename)
        self.assertTrue(
            os.path.exists(filename), "write_file did not actually write a file"
        )
        os.remove(filename)

    def test_recurse_unused(self):
        """
        Tests against an empty outspec.
        """

        CS = CheckScript(None, self.script1dict)
        checktext = CS.recurse(CS.outspec, CS.specs_from_file, pretty_print=True)
        self.assertTrue("WARNING 1" in checktext)

    def test_recurse_unspecified(self):
        """
        Tests against an empty input spec
        """

        CS = CheckScript(None, self.script1dict)
        checktext = CS.recurse(CS.specs_from_file, CS.outspec, pretty_print=True)
        self.assertTrue("WARNING 2" in checktext)

    def test_recurse_both_use(self):
        """
        Tests template_completeness_testing against template_prototype_testing
        """

        CS = CheckScript(self.script2path, self.script3dict)
        checktext = CS.recurse(CS.specs_from_file, CS.outspec, pretty_print=True)
        self.assertTrue("WARNING 3" in checktext)

    def test_recurse_warn_4(self):

        CS = CheckScript(self.script1path, self.script2dict)
        checktext = CS.recurse(CS.specs_from_file, CS.outspec, pretty_print=True)
        self.assertTrue("WARNING 4" in checktext)


if __name__ == "__main__":
    unittest.main()
