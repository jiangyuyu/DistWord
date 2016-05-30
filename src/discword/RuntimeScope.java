package discword;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.*;

public class RuntimeScope {
	HashMap<String, List<Pair>> map = new HashMap<String, List<Pair>>();
	boolean Create(String name) {
		if (map.containsKey(name)) return false;
		map.put(name, new ArrayList<Pair>());
		return true;
	}
	
	boolean Add(String name, int it, Comparable value) {
		if (!map.containsKey(name)) return false;
		map.get(name).add(new Pair(it,value));
		return true;
	}
	
	void WriteTo(String filename) {
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			writer.write("---RuntimeScope info---\n");
			for (String key : map.keySet()) {
				List<Pair> data = map.get(key);
				writer.write(String.format("key=%s size=%d\n", key, data.size()));
				for (Pair p : data) {
					writer.write(String.format("%d %s\n", p.first, p.second));
				}
				writer.write("\n");
			}
			writer.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
