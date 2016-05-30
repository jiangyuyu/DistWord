package discword;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class TopicDiscriminativeModel extends Model {
	// class variables
	public static String topicSummarySuffix;
	public static String wordSummarySuffix;

	// model variables
	
	// temporary variables
	public int NN;
	public int[] NW;
	public double [][] reversePhi;	// p(z|w) matrix K*V
	public double [] topicProb;		// p(z|Corpus)
	public double [] TK;  // topic knowledge of words
	
	protected double [][] nw; 	// nw[i][j]: number of instances of word/term i assigned to topic j, size V * K
	protected double [][] nd; 	// nd[i][j]: number of words in document i assigned to topic j, size M x K
	protected double [] nwsum;  // nwsum[j]: total number of words assigned to topic j, size K
	protected double [] ndsum;  // ndsum[i]: total number of words in document i, size M

	// experiment setting
	public boolean dryRun = false;
	
	public TopicDiscriminativeModel() {
		super();
	}
	
	public void setDefaultValues() {
		nw = null;
		nd = null;
		nwsum = null;
		ndsum = null;
		reversePhi = null;
		topicProb = null;
		TK = null;
	}
	
	public boolean saveModel(String modelName){
		if (twords > 0){
			if (!saveModelTopicSummary(dir + File.separator + modelName + topicSummarySuffix))
				return false;
			if (!saveModelWordSummary(dir + File.separator + modelName + wordSummarySuffix))
				return false;
		}
		return super.saveModel(modelName);
	}
	
	public boolean saveModelTopicSummary(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream(filename), "UTF-8"));
			
			if (twords > V){
				twords = V;
			}
			
			for (int k = 0; k < K; k++){
				//print topic
				writer.write(String.format("Topic %dth - probability=%.6f:\n", k, topicProb[k]));
				
				// top words of probability
				List<Pair> wordsProbsList = new ArrayList<Pair>(); 
				for (int w = 0; w < V; w++){
					Pair p = new Pair(w, phi[k][w], false);
					wordsProbsList.add(p);
				}//end foreach word
				Collections.sort(wordsProbsList);
				
				// top words of entropy based weights prob * deterministic
				List<Pair> wordsWeightsList = new ArrayList<Pair>(); 
				for (int w = 0; w < V; w++){
					Pair p = new Pair(w, reversePhi[k][w] * phi[k][w], false);
					wordsWeightsList.add(p);
				}//end foreach word
				Collections.sort(wordsWeightsList);
				
				for (int i = 0; i < twords; i++){
					if (data.dict.contains((Integer)wordsProbsList.get(i).first)){
						String word1 = data.dict.getWord((Integer)wordsProbsList.get(i).first);
						int w1 = (Integer)wordsProbsList.get(i).first;
						
						String word2 = data.dict.getWord((Integer)wordsWeightsList.get(i).first);
						int w2 = (Integer)wordsWeightsList.get(i).first;
						
						writer.write(String.format("\t%16s %.6f %.6f %.6f\t%16s %.6f %.6f %.6f\n",
								word1, phi[k][w1], reversePhi[k][w1], phi[k][w1] * reversePhi[k][w1],
								word2, phi[k][w2], reversePhi[k][w2], phi[k][w2] * reversePhi[k][w2]));
					}
				}
			} //end foreach topic
						
			writer.close();
		}
		catch(Exception e){
			System.out.println("Error while saving model twords: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	public boolean saveModelWordSummary(String filename) {
		try{
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream(filename), "UTF-8"));
			
			Pair[] wordList = new Pair[V];
			for (int w = 0; w < V; w++){
				wordList[w] = new Pair(w, TK[w] * NW[w], false);
			}
			Arrays.sort(wordList);
			for (int i=0; i<V/10; i++) {
				int w = (Integer)wordList[i].first;
				String word = data.dict.getWord(w);
				writer.write(String.format("%16s %6d %6f %.6f %.6f\n", word, NW[w], TK[w], wordList[i].second, Util.GiniSparsity(nw[w])));
			}
			writer.write("\n\n\n\n\n");
			
			for (int w = 0; w < V; w++){
				wordList[w] = new Pair(w, TK[w], true);
			}
			Arrays.sort(wordList);
			for (int i=0; i<V/10; i++) {
				int w = (Integer)wordList[i].first;
				String word = data.dict.getWord(w);
				writer.write(String.format("%16s %6d %6f %.6f %.6f\n", word, NW[w], TK[w], TK[w] * NW[w], Util.GiniSparsity(nw[w])));
			}
			
			writer.close();
		}
		catch(Exception e){
			System.out.println("Error while saving model twords: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}

	protected boolean initParameters(LDACmdOption option){
		if (!super.initParameters(option)) return false;
		dryRun = option.entropyDryRun;
		return true;
	}
	
	protected boolean initDataset(LDACmdOption option, Dictionary globalDict) {
		if (!super.initDataset(option, globalDict)) return false;
		NN = 0;
		NW = new int[V];
		for (int d=0; d<M; d++) {
			for (int w : data.docs[d].words) {
				NW[w] ++;
			}
			NN += data.docs[d].words.length;
		}
		return true;
	}
	
	protected boolean initZeroProbs() {
		if (!super.initZeroProbs()) return false;
		reversePhi = new double[K][V];
		topicProb = new double[K];
		TK = new double[V];
		Arrays.fill(TK, 1.0);
		return true;
	}
	
	public boolean initZeroCounters() {
		if (!super.initZeroCounters()) return false;
		int w, k, m;
		nw = new double[V][K];
		for (w = 0; w < V; w++){
			for (k = 0; k < K; k++){
				nw[w][k] = 0;
			}
		}
		
		nd = new double[M][K];
		for (m = 0; m < M; m++){
			for (k = 0; k < K; k++){
				nd[m][k] = 0;
			}
		}
		
		nwsum = new double[K];
		for (k = 0; k < K; k++){
			nwsum[k] = 0;
		}
		
		ndsum = new double[M];
		for (m = 0; m < M; m++){
			ndsum[m] = 0;
		}
		
		return true;
	}
	
	public void computeReversePhi() {
		double betaSum = beta * K;
		for (int w=0; w<V; w++) {
			double total = Util.sum(nw[w]) + betaSum;
			for (int k=0; k<K; k++) {
				reversePhi[k][w] = (nw[w][k] +beta) / total;
			}
		}
	}
	
	public void computeTopicProb() {
		double total = Util.sum(nwsum);
		for (int k=0; k<K; k++) {
			topicProb[k] = nwsum[k] / total;
		}
	}
	
	public void computeWordInfoRate() {
		for (int w = 0; w < V; w++) {
			double total = Util.sum(nw[w]) + beta * K;
			for (int k = 0; k < K; k++) {
				assert nw[w][k] >= 0.0 : String.format("nw < 0 - %d-%d-%f", w,
						k, nw[w][k]);
				p[k] = (nw[w][k] + beta) / total;
			}
			TK[w] = Util.infoRate(p);
		}
	}
}
