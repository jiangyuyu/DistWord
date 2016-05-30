
package discword;

import java.io.File;
import java.util.Arrays;

public class EntropyEstimator extends Estimator {
	
	// output model
	protected EntropyModel trnModel;
	LDACmdOption option;
	
	CircularZ cz = null;
	int updateInterval = 5;
	boolean dryRun = false;
	
	public boolean init(LDACmdOption option){
		this.option = option;
		trnModel = new EntropyModel();
		
		if (option.jobType == JobType.ESTW){
			if (!trnModel.initEstModel(option))
				return false;
			if (!option.useDict) {
				trnModel.data.dict.writeWordMap(option.dir + File.separator + option.modelName + "." + option.wordMapFileName);
			}
			
			// init CZ
			int NN = 0;
			for (int d = 0; d < trnModel.M; d++) {
				NN += trnModel.z[d].size();
			}
			cz = new CircularZ(NN);
			for (int d = 0; d < trnModel.M; d++) {
				for (int n = 0; n < trnModel.z[d].size(); n++) {
					int topic = trnModel.z[d].get(n);
					int w = trnModel.data.docs[d].words[n];
					cz.add(trnModel.TK[w]);
				}
			}
			
			if (option.entropyDryRun) {
				System.out.println("-------------entropy model dry run------------------");
				dryRun = true;
			}
		}
		else {
			assert(false); // not supported
		}
		return true;
	}
	
	public void estimate(){
		System.out.println("Sampling " + option.niters + " iteration!");
		
		int lastIter = trnModel.liter;
		for (trnModel.liter = lastIter + 1; trnModel.liter < option.niters + lastIter; trnModel.liter++){			
			// for all z_i
			for (int m = 0; m < trnModel.M; m++){
				for (int n = 0; n < trnModel.data.docs[m].length; n++){
					sampling(m, n);
				} // end for each word
			} // end for each document
			
			assert Util.doubleEquals(Util.sum(trnModel.nwsum), Util.sum(trnModel.ndsum)) : 
				String.format("%.10f == %.10f", Util.sum(trnModel.nwsum), Util.sum(trnModel.ndsum)); 
			System.out.println(String.format("%d - total sampling size = %.6f", trnModel.liter, Util.sum(trnModel.ndsum)));
			
			// periodically update word info ratio
			if (trnModel.liter % updateInterval == 0) {
				trnModel.computeWordInfoRate();
				System.out.println(String.format("%d - sample-sum=%.6f perplexity = %.6f", 
						trnModel.liter, Util.sum(trnModel.ndsum), 
						trnModel.perplexity()));
			}
			
			if (option.savestep > 0){
				if (trnModel.liter % option.savestep == 0){
					System.out.println("Saving the model at iteration " + trnModel.liter + " ...");
					computeTheta();
					computePhi();
					trnModel.computeWordInfoRate();
					trnModel.computeTopicProb();
					trnModel.computeReversePhi();
					trnModel.saveModel("model-" + Conversion.ZeroPad(trnModel.liter, 5));
				}
			}
		}// end iterations		
		
		System.out.println("Gibbs sampling completed!\n");
		System.out.println("Saving the final model!\n");
		computeTheta();
		computePhi();
		trnModel.computeWordInfoRate();
		trnModel.computeTopicProb();
		trnModel.computeReversePhi();
		trnModel.liter--;
		trnModel.saveModel("model-final");
	}
	
	/**
	 * Do sampling
	 * @param m document number
	 * @param n word number
	 * @return topic id
	 */
	public int sampling(int m, int n){
		// remove z_i from the count variable
		int old_topic = trnModel.z[m].get(n);
		int w = trnModel.data.docs[m].words[n];
		assert w < trnModel.V : String.format("m=%d n=%d z=%d w=%d", m, n, old_topic, w);
		
		double oldWeight = cz.get();
		Util.safeSubstract(trnModel.nw, w, old_topic, oldWeight);
		Util.safeSubstract(trnModel.nd, m, old_topic, oldWeight);
		Util.safeSubstract(trnModel.nwsum, old_topic, oldWeight);
		Util.safeSubstract(trnModel.ndsum, m, oldWeight);
		
		double Vbeta = trnModel.V * trnModel.beta;
		double Kalpha = trnModel.K * trnModel.alpha;
		
		// do multinominal sampling via cumulative method
		for (int k = 0; k < trnModel.K; k++){
			trnModel.p[k] = (trnModel.nw[w][k] + trnModel.beta)/(trnModel.nwsum[k] + Vbeta) *
					(trnModel.nd[m][k] + trnModel.alpha)/(trnModel.ndsum[m] + Kalpha);
			assert (!Double.isNaN(trnModel.p[k]) && trnModel.p[k] > 0.0) : "invalid new p(z|w) " + trnModel.p[k];
		}
		
		// cumulate multinomial parameters
		for (int k = 1; k < trnModel.K; k++){
			trnModel.p[k] += trnModel.p[k - 1];
		}
		
		// scaled sample because of unnormalized p[]
		double u = Util.random() * trnModel.p[trnModel.K - 1];
		
		int new_topic = -1;
		for (new_topic = 0; new_topic < trnModel.K; new_topic++){
			if (trnModel.p[new_topic] > u) //sample topic w.r.t distribution p
				break;
		}
		
		assert new_topic < trnModel.K : Arrays.toString(trnModel.p);
		if (new_topic == trnModel.K) {
			System.out.println();
		}
		
		// add newly estimated z_i to count variables
		double newWeight = dryRun ? 1.0 : trnModel.TK[w];
		assert newWeight > 0.0;
		cz.add(newWeight);
		trnModel.nw[w][new_topic] += newWeight;
		trnModel.nd[m][new_topic] += newWeight;
		trnModel.nwsum[new_topic] += newWeight;
		trnModel.ndsum[m] += newWeight;
		
		trnModel.z[m].set(n, new_topic);
 		return new_topic;
	}
	
	public void computeTheta(){
		for (int m = 0; m < trnModel.M; m++){
			for (int k = 0; k < trnModel.K; k++){
				trnModel.theta[m][k] = (trnModel.nd[m][k] + trnModel.alpha) / (trnModel.ndsum[m] + trnModel.K * trnModel.alpha);
			}
		}
	}
	
	public void computePhi(){
		for (int k = 0; k < trnModel.K; k++){
			for (int w = 0; w < trnModel.V; w++){
				trnModel.phi[k][w] = (trnModel.nw[w][k] + trnModel.beta) / (trnModel.nwsum[k] + trnModel.V * trnModel.beta);
			}
		}
	}
}
