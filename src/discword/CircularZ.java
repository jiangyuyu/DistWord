package discword;

public class CircularZ {
	private double[] buffer;
	private int offset; // always point to new position (and with old data)
	
	public CircularZ(int NN) {
		buffer = new double[NN];
		offset = 0;
	}
	
	public double get() {
		return buffer[offset];
	}
	
	public void add(double value) {
		buffer[offset] = value;
		offset = (offset + 1) % buffer.length;
	}
}
