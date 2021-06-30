# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import numpy
import pandas
from pyspark.ml.feature import ElementwiseProduct
from pyspark.ml.linalg import Vectors
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmlElementwiseProduct(SparkMlTestCase):

    @unittest.skipIf(sys.version_info < (3, 8),
                     reason="pickle fails on python 3.7")
    def test_element_wise_product(self):
        data = self.spark.createDataFrame([(Vectors.dense([2.0, 1.0, 3.0]),)], ["features"])
        model = ElementwiseProduct(scalingVec=Vectors.dense([1.0, 2.0, 3.0]),
                                   inputCol="features", outputCol="eprod")
        feature_count = data.first()[0].size
        model_onnx = convert_sparkml(model, 'Sparkml ElementwiseProduct',
                                     [('features', FloatTensorType([None, feature_count]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        expected = [
            predicted.toPandas().eprod.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
            ]
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = numpy.vstack([data_np, data_np])
        paths = save_data_models(data_np, expected, model, model_onnx, basename="SparkmlElementwiseProduct")
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(['eprod'], data_np, onnx_model_path)
        print(output[0])
        print(expected[0])
        expected = numpy.vstack([expected[0], expected[0]])
        compare_results(expected, output[0], decimal=5)


if __name__ == "__main__":
    unittest.main()
