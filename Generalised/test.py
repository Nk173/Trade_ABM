import unittest
from init import case, countries, count, industries, P, A, alpha, beta
from main import gulden
import numpy as np

class TestGulden(unittest.TestCase):
    def test_exports(self):
        g1 = gulden()
        diagexports = np.zeros((len(countries),1))
        ref = np.zeros((len(countries),1))

        for c in range(len(countries)):
            diagexports[c] = g1[countries[c]][industries[c]]
            ref[c] = 190
        self.assertGreaterEqual(diagexports.all(), ref.all(), 'Test for comparative advantage violated!')

if __name__=='__main__':
    unittest.main()



