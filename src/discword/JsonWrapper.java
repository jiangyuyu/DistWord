package discword;


import java.io.BufferedInputStream;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class JsonWrapper {
	public static JSONArray jsonArray(String source){
		try {
			JSONArray jarray = new JSONArray(source);
			return jarray;
		} catch (JSONException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public static JSONObject jsonMap(String source){
		try {
			JSONObject jmap = new JSONObject(source);
			return jmap;
		} catch (JSONException e) {
			e.printStackTrace();
		}
		return null;	
	}
	
	public static JSONObject config(String source){
		try {
			JSONObject jmap = new JSONObject(source);
			return jmap;
		} catch (JSONException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public static void main(String[] args){
		String source = "{\"tpp\" : [[1,2], [3,4], [5,6]], \"K\" : 10}";
		JSONObject cfg = config(source);
		try {
			JSONArray tpp = cfg.getJSONArray("tpp");
			Object obj = cfg.get("tpp");
			if (obj.getClass() == JSONArray.class){
				System.out.println("123");
			}
			int k = cfg.getInt("K");
			for (int i=0; i<tpp.length(); i++){
				JSONArray item = tpp.getJSONArray(i);
				System.out.println(item);
			}
			System.out.println(k);
		} catch (JSONException e) {
			e.printStackTrace();
		}		
	}
}
