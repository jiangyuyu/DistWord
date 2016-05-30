
package discword;

import java.io.File;
import java.util.Arrays;

public class WeightedEstimator extends Estimator {
	
	// output model
	protected WeightedModel trnModel;
	LDACmdOption option;
	
	int updateWeightInterval = 5;
	double[][] old_nw, cur_nw;
	double[][] hist_nw;
	
	EvolveSampleContext context = null;
	int lastWord = 0;
	CircularZ cz = null;
	
	public boolean init(LDACmdOption option){
		this.option = option;
		trnModel = new WeightedModel();
		
		if (option.jobType == JobType.ESTW){
			if (!trnModel.initEstModel(option))
				return false;
			trnModel.data.dict.writeWordMap(option.dir + File.separator + option.modelName + "." + option.wordMapFileName);
			
			context = new EvolveSampleContext(trnModel.V);
			lastWord = trnModel.data.docs[trnModel.M - 1].length - 1;
			
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
					cz.add(trnModel.vnw[w][topic]);
				}
			}
			
			// init old_nw, cur_nw
			old_nw = new double[trnModel.V][trnModel.K];
			cur_nw = new double[trnModel.V][trnModel.K];
			Util.copy2D(trnModel.nw, old_nw);
			Util.fill2D(cur_nw, 0.0);
		}
		else {
			// not supported
			assert(false);
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
					int topic = sampling(m, n);
					trnModel.z[m].set(n, topic);
				} // end for each word
			} // end for each document
			
			Util.add2D(cur_nw, trnModel.nw);
			
			// update weights on last word in last document in satisfied iteration.
			if ((trnModel.liter % updateWeightInterval) == 0) {
				System.out.println("update weight " + trnModel.liter);
				Util.mupltiplyScalar(cur_nw, 1.0 / updateWeightInterval);
				context.computeWeights(old_nw, cur_nw, trnModel.vnw);
				Util.copy2D(cur_nw, old_nw);
				System.out.println();
			}
			
			if (option.savestep > 0){
				if (trnModel.liter % option.savestep == 0){
					System.out.println("Saving the model at iteration " + trnModel.liter + " ...");
					computeTheta();
					computePhi();
					trnModel.saveModel("model-" + Conversion.ZeroPad(trnModel.liter, 5));
				}
			}
		}// end iterations		
		
		System.out.println("Gibbs sampling completed!\n");
		System.out.println("Saving the final model!\n");
		computeTheta();
		computePhi();
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
		int topic = trnModel.z[m].get(n);
		int w = trnModel.data.docs[m].words[n];
		
		trnModel.nw[w][topic] -= cz.get();
		trnModel.nd[m][topic] -= 1;
		trnModel.nwsum[topic] -= cz.get();
		trnModel.ndsum[m] -= 1;
		
		assert trnModel.nw[w][topic] >= 0.0 : String.format("%d-%d-%d %d-%d=%f=%f",
				trnModel.liter, m, n, topic, w, trnModel.vnw[w][topic], trnModel.nw[w][topic]);
		
		double Vbeta = trnModel.V * trnModel.beta;
		double Kalpha = trnModel.K * trnModel.alpha;
		
		// do multinominal sampling via cumulative method
		for (int k = 0; k < trnModel.K; k++){
			trnModel.p[k] = (trnModel.nw[w][k] + trnModel.beta)/(trnModel.nwsum[k] + Vbeta) *
					(trnModel.nd[m][k] + trnModel.alpha)/(trnModel.ndsum[m] + Kalpha);
		}
		
		// cumulate multinomial parameters
		for (int k = 1; k < trnModel.K; k++){
			trnModel.p[k] += trnModel.p[k - 1];
		}
		
		// scaled sample because of unnormalized p[]
		double u = Util.random() * trnModel.p[trnModel.K - 1];
		
		for (topic = 0; topic < trnModel.K; topic++){
			if (trnModel.p[topic] > u) //sample topic w.r.t distribution p
				break;
		}
		
		// add newly estimated z_i to count variables
		double newWeight = trnModel.vnw[w][topic];
		newWeight = 1.0;
		cz.add(newWeight);
		trnModel.nw[w][topic] += newWeight;
		trnModel.nd[m][topic] += 1;
		trnModel.nwsum[topic] += newWeight;
		trnModel.ndsum[m] += 1;
		
 		return topic;
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
