package discword;

import java.util.ArrayList;

public class ModelHelper {
	public static double KLDivergence(Model m1, Model m2) {
		int K = m1.K, V = m1.V;
		ArrayList<Double> values = new ArrayList<Double>();
		for (int i=0; i<K; i++) {
			for (int j=0; j<K; j++) {
				//m1[i] -> m2[j] TODO
			}
		}
		return 0.0;
	}
}
