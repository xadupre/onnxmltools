# SPDX-License-Identifier: Apache-2.0

"""
Tests CoreML Scaler converter.
"""
import unittest
from distutils.version import StrictVersion
import numpy
try:
    from sklearn.impute import SimpleImputer as Imputer
    import sklearn.preprocessing
    if not hasattr(sklearn.preprocessing, 'Imputer'):
        # coremltools 3.1 does not work with scikit-learn 0.22
        setattr(sklearn.preprocessing, 'Imputer', Imputer)
except ImportError:
    from sklearn.preprocessing import Imputer
try:
    import coremltools
except ImportError:
    coremltools = None
from sklearn.preprocessing import StandardScaler
from onnxmltools.convert.coreml.convert import convert
from onnxmltools.utils import dump_data_and_model


class TestCoreMLScalerConverter(unittest.TestCase):

<<<<<<< HEAD
    @unittest.skipIf(coremltools is None, "coremltools not available")
=======
    @unittest.skipIf(
        StrictVersion(coremltools.__version__) > StrictVersion("3.1"),
        reason="untested")
>>>>>>> cb2782b155ff67dc1e586f36a27c5d032070c801
    def test_scaler(self):
        model = StandardScaler()
        data = numpy.array([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]], dtype=numpy.float32)
        model.fit(data)
        model_coreml = coremltools.converters.sklearn.convert(model)
        model_onnx = convert(model_coreml.get_spec())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="CmlStandardScalerFloat32")


if __name__ == "__main__":
    unittest.main()
