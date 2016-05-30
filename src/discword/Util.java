package discword;

import java.util.Arrays;

public class Util {
	static java.util.Random random = new java.util.Random(123456);
	
	static double random() {
		return random.nextDouble();
	}
	
	static boolean doubleEquals(double a, double b) {
		return Math.abs(a - b) <= 1e-6;
	}
	
	static void copy(double[] src, double[] dest) {
	    System.arraycopy(src, 0, dest, 0, src.length);
	}
	
	static void copy2D(double[][] src, double[][] dest) {
		int size = src[0].length;
		for (int i=0; i<src.length; i++) {
			System.arraycopy(src[i], 0, dest[i], 0, size);
		}
	}
	
	static void fill2D(double[][] src, double val) {
		int size = src[0].length;
		for (int i=0; i<src.length; i++) {
			Arrays.fill(src[i], val);
		}
	}
	
	static void mupltiplyScalar(double[][] src, double x) {
		for (double[] row : src) {
			for (int j=0; j<row.length; j++)
				row[j] *= x;
		}
	}
	
	static void add2D(double[][] src, double[][] x) {
		for (int i=0; i<src.length; i++) {
			for (int j=0; j<src[i].length; j++)
				src[i][j] += x[i][j];
		}
	}
	
	static double entropy(double[] p) {
		double value = 0.0;
		double total = sum(p);
		for (double x : p) {
			if (x == 0.0) continue; 
			value -= x / total * Math.log(x / total);
		}
		return value;
	}
	
	static double infoRate(double[] p) {
		if (p == null || p.length == 0) return 1.0;
		double ret = (1.0 - entropy(p) / Math.log(p.length));
		assert (ret >= -1e-10 && ret <= 1.0 + 1e-10) : 
			String.format("inforate=%f, entropy(p) = %f, len(p) = %d, p = %s", ret, entropy(p), p.length, Arrays.toString(p));
		if (ret < 1e-10) ret = 1e-10;
		assert ret > 0.0 && ret <= 1.0 : String.format("invalid infoRate: %.6f - %s", ret, Arrays.toString(p));
		return ret;
	}
	
	static double sum(double[] p) {
		double value = 0.0;
		for (double x : p) value += x;
		return value;
	}
	
	static double sumCol(double[][] arr, int col) {
		double value = 0.0;
		for (double[] x : arr) {
			value += x[col];
		}
		return value;
	}
	
	static void safeSubstract(double[][] arr, int i, int j, double sub) {
		assert arr[i][j] - sub > -1e-10 : "safe substract error: " + arr[i][j] + " " + sub;
		if (arr[i][j] - sub < 0.0) arr[i][j] = 0.0;
		else arr[i][j] -= sub;
	}
	
	static void safeSubstract(double[] arr, int i, double sub) {
		assert arr[i] - sub > -1e-10 : "safe substract error: " + arr[i] + " " + sub;
		if (Math.abs(arr[i] - sub) < 0.0) arr[i] = 0.0;
		else arr[i] -= sub;
	}
	
	static double HoyerSparsity(double[] arr) {
		if (arr == null || arr.length == 1) return 1.0;
		double sqrtN = Math.sqrt(arr.length);
		double x = 0.0, y = 0.0;
		for (double z : arr) {
			x += z;
			y += z * z;
		}
		return (sqrtN - x / Math.sqrt(y)) / (sqrtN - 1);
	}
	
	/* 
	 * 2 * sigma_i {i*y_i} / (n * sigma_i {y_i}) - (n + 1) / n
	 */
	static double GiniSparsity(double[] arr) {
		double[] y = Arrays.copyOf(arr, arr.length);
		Arrays.sort(y);
		double a = 0.0, b = 0.0, n = y.length;
		for (int i=1; i<=y.length; i++) {
			a += i * y[i-1];
			b += y[i-1];
		}
		return ((2 * a) / (n * b) - (n + 1) / n);
	}
	
	static double KLDivergence(double[] p, double[] q) {
		double value = 0.0;
		for (int i=0; i<p.length; i++) {
			value += p[i] * Math.log(p[i] / q[i]);
		}
		return value;
	}
	
	static Pair AvgStdDev(double[] arr) {
		if (arr == null || arr.length == 0) return null;
		if (arr.length == 1) return new Pair(arr[0], 0.0);
		double avg = sum(arr) / arr.length;
		double total = 0.0;
		for (double a : arr) {
			total += (a - avg) * (a - avg);
		}
		double stdev = Math.sqrt(total / (arr.length - 1));
		return new Pair(avg, stdev);
	}
}
