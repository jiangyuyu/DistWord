package discword;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class TEConfig {
	protected JSONObject cfg;
	protected int npriors = -1;
	protected int[][] partition = null;
	protected int[][] priorTopics = null;
	
	public TEConfig(String fpath){
		try {
			cfg = new JSONObject(FileUtil.read(fpath));
			npriors = _getNumberOfTopicPriors();
			partition = _getPartition();
			priorTopics = _getPriorTopics();
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}
	
	public int[] getPartition(int slot){
		return partition[slot];
	}
	
	private int[][] _getPartition(){
		try {
			JSONArray partition = cfg.getJSONArray("ptt");
			int[][] ptt = new int[partition.length()][2];
			for (int i=0; i<partition.length(); i++){
				JSONArray slot = partition.getJSONArray(i);
				ptt[i][0] = slot.getInt(0);
				ptt[i][1] = slot.getInt(1);
			}
			return ptt;
		} catch (JSONException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public int getNumberOfTopicPriors(){
		return npriors;
	}
	
	private int _getNumberOfTopicPriors(){
		int nPriors = -1;
		try {
			nPriors = cfg.getInt("npriors");
		} catch (JSONException e) {
			e.printStackTrace();
		}
		return nPriors;
	}
	
	public int[] getPriorTopics(int slot){
		return priorTopics[slot];
	}
	
	private int[][] _getPriorTopics(){
		if (npriors == -1) return null;
		try {
			JSONArray topics = cfg.getJSONArray("prior_topics");
			int[][] allTopics = new int[topics.length()][npriors];
			for (int i=0; i<topics.length(); i++){
				JSONArray localTopics = topics.getJSONArray(i);
				for (int j=0; j<localTopics.length(); j++){
					allTopics[i][j] = localTopics.getInt(j);
				}
			}
			return allTopics;
		} catch (JSONException e) {
			e.printStackTrace();
		}
		return null;
	}
}
