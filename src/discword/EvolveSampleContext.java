package discword;

import java.util.ArrayList;
import java.util.Arrays;

public class EvolveSampleContext {
	public static double FocusVolumneDF = Feature.CONSTANT;
	public static double FocusWeightDF = Feature.CONSTANT;
	public static double IntervalDF = Feature.CONSTANT;
	
	public int Generation = 0;		// current iteration generation
	public int LastGeneration = 0;
	public int Volumne = 0;
	
	public int Interval;
	
	public double FocusVolumne;	// x * delta
	public double UpWeight;
	public double DownWeight;
	public double UpdateRatio;
	
	private double [] diff = null;
	
	public EvolveSampleContext(int V) {
		assert(V > 0);
		Volumne = V;
		diff = new double[Volumne];
		
		FocusVolumne = 2.0;
		UpWeight = 2.0;
		DownWeight = 0.5;
		UpdateRatio = 0.1;	
	}
	
	public void iteration() {
		Generation ++;
	}
	
	public boolean needsUpdate() {
		return (Generation >= LastGeneration + interval());
	}
	public double interval() {
		return Interval;
	}
	
	public void computeW2(double[][][] hist, double[][] nw, double[][] vnw) {
		int V = nw.length, K = nw[0].length, iters = hist.length;
		double[] xx = new double[iters];
		
		ArrayList<Item> q = new ArrayList<Item>();
		for (int w=0; w<V; w++) {
			for (int k=0; k<K; k++) {
				for (int i=0; i<iters; i++) xx[i] = hist[i][w][k];
				Stat stat = computeStatistics(xx);
				if (Math.abs(nw[w][k] - stat.mu) > focusVolumne() * stat.delta) {
					q.add(new Item(w, k, nw[w][k] > stat.mu ? 1 : -1));
				}
			}
		}
		
		System.out.print(q.size()/K + "  ");
		for (Item x : q) {
			vnw[x.index][(int) Math.round(x.value)] *= (x.sgn == 1) ? upWeight() : downWeight();
		}
	}
	
	public void computeWeights(double[][] a, double[][] b, double[][] vnw) {
		assert(a.length == Volumne && b.length == Volumne && vnw.length == Volumne);
		// Forcus terms are outlier terms with biggest changes.
		
		for (int k = 0; k < a[0].length; k++) {
			double suma = 0.0, sumb = 0.0;
			for (int i=0; i<Volumne; ++i) {
				suma += a[i][k];
				sumb += b[i][k];
			}
			
			for (int i=0; i<Volumne; ++i) {
				diff[i] = b[i][k]/sumb - a[i][k]/suma; // normalized setting
				//vnw[i][k] = 1.0;
			}
			
			Stat stat = computeStatistics(diff);
			
			ArrayList<Item> q = new ArrayList<Item>();
			for (int i=0; i<Volumne; ++i) {
				if ((Math.abs(diff[i] - stat.mu) > focusVolumne() * stat.delta) &&
						(Math.abs(diff[i]) >= UpdateRatio * a[i][k] / suma)) {
					q.add(new Item(i, Math.abs(diff[i]), diff[i] > stat.mu ? 1 : -1));
				}
			}
			
			System.out.print(q.size() + "    ");
			
			for (Item x : q) {
				//vnw[x.index][k] *= (x.sgn == 1) ? upWeight() : downWeight();
				vnw[x.index][k] *= b[x.index][k] / a[x.index][k];
			}
		}
	}
	
	public double focusVolumne() {
		return FocusVolumne;
	}
	
	public double upWeight() {
		return UpWeight;
	}
	public double downWeight() {
		return DownWeight;
	}
	
	private static class Item {
		public int index;
		public double value;
		int sgn;
		public Item(int i, double v, int sgn) {
			index = i;
			value = v;
			this.sgn = sgn;
		}
	}
	
	public static class Stat {
		double mu = 0.0;
		double delta = 0.0;
	}
	public static Stat computeStatistics(double[] a) {
		if (a == null || a.length == 0) return null;
		Stat stat = new Stat();
		if (a.length == 1) {
			stat.mu = a[0];
			return stat;
		}
		
		double total = 0.0;
		for (double x : a) total += x;
		stat.mu= total / a.length;
		
		double delta = 0.0;
		for (double x : a) delta += Math.pow(x - stat.mu, 2);
		stat.delta = Math.sqrt(delta / (a.length - 1));
		
		return stat;
	}
	
	public static class Feature {
		public static final double CONSTANT = -1.0;
		public double Value = 0.0;
		public double DamplingFactor = CONSTANT; 	// feature.Value *= feature.DamplingFactor, <= 0 - constant feature
		public double ActionInterval = CONSTANT;	// <= 0 - constant feature in evolving process. > 0 - changing feature
		public double AIDamplingFactor = CONSTANT;	// <= 0 - constant action interval
		public int LastTime = 0;
		public int CurrentTime = 0;
		
		public Feature(double value) {
			Value = value;
		}
		
		public Feature(double value, double df, double ai, double ai_df) {
			Value = value;
			DamplingFactor = df;
			ActionInterval = ai;
			AIDamplingFactor = ai_df;
		}
		
		public double act() {
			CurrentTime ++;
			if (ActionInterval <= 0.0 && DamplingFactor > 0.0 && (CurrentTime - LastTime) >= ActionInterval) {
				Value *= DamplingFactor;
				LastTime = CurrentTime;
				if (AIDamplingFactor > 0.0) ActionInterval *= AIDamplingFactor;
			}
			return Value;
		}
	}
	
	public static boolean doubleEQ(double a, double b) {
		return Math.abs(a - b) < 1.0e-6;
	}
	
	public static boolean doubleGT(double a, double b) {
		return (a > b + 1.0e-6);
	}
}
