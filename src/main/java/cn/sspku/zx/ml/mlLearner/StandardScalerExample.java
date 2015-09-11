package cn.sspku.zx.ml.mlLearner;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.feature.StandardScaler;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import scala.Tuple2;

public class StandardScalerExample {

	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("StandardScaler");
		SparkContext sc = new SparkContext(conf);

		JavaRDD<LabeledPoint> data = MLUtils
				.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
				.toJavaRDD().cache();

		@SuppressWarnings("serial")
		final StandardScalerModel scaler1 = new StandardScaler().fit(data.map(
				new Function<LabeledPoint, Vector>() {
					public Vector call(LabeledPoint p) throws Exception {
						// TODO Auto-generated method stub
						return p.features();
					}
				}).rdd());

		@SuppressWarnings("serial")
		final StandardScalerModel scaler2 = new StandardScaler(true, true)
				.fit(data.map(new Function<LabeledPoint, Vector>() {
					public Vector call(LabeledPoint p) throws Exception {
						// TODO Auto-generated method stub
						return p.features();
					}
				}).rdd());

		// data1 will be unit variance.
		@SuppressWarnings("serial")
		JavaRDD<Tuple2<Double, Vector>> data1 = data
				.map(new Function<LabeledPoint, Tuple2<Double, Vector>>() {

					public Tuple2<Double, Vector> call(LabeledPoint x)
							throws Exception {
						return new Tuple2<Double, Vector>(x.label(), scaler1
								.transform(x.features()));
					}
				});
		System.out.println(data1);

		// Without converting the features into dense vectors, transformation
		// with zero mean will raise
		// exception on sparse vector.
		// data2 will be unit variance and zero mean.
		@SuppressWarnings("serial")
		JavaRDD<Tuple2<Double, Vector>> data2 = data
				.map(new Function<LabeledPoint, Tuple2<Double, Vector>>() {
					public Tuple2<Double, Vector> call(LabeledPoint x)
							throws Exception {
						return new Tuple2<Double, Vector>(x.label(), scaler2
								.transform(Vectors
										.dense(x.features().toArray())));
					}
				});
		System.out.println(data2);

	}
}
