package discword;

import java.util.Arrays;

public class MockTDModel extends TopicDiscriminativeModel {
	public void computeWordInfoRate() {
		Arrays.fill(TK, 1.0);
	}
}
