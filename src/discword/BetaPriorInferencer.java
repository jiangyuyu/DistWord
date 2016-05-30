package discword;

import java.io.File;

public class BetaPriorInferencer extends Inferencer {
	double[][] bp = null;  // beta prior vector V*K
	double[] bpsum = null; // K
	TEConfig cfg;
	
	@Override
	protected int infSampling(int m, int n) {
		int topic=-1, w=-1;
		
		topic = newModel.z[m].get(n);
		w= newModel.data.docs[m].words[n];
		newModel.nw[w][topic] -= 1;
		newModel.nd[m][topic] -= 1;
		newModel.nwsum[topic] -= 1;
		newModel.ndsum[m] -= 1;
		
		double Kalpha = newModel.K * newModel.alpha;
		double Vbeta = newModel.V * newModel.beta;
		
		// do multinomial sampling via cummulative method
		for (int k = 0; k < newModel.K; k++){			
			newModel.p[k] = (newModel.nw[w][k] + bp[w][k] + newModel.beta)/(newModel.nwsum[k] + bpsum[k] + Vbeta) *
					(newModel.nd[m][k] + newModel.alpha)/(newModel.ndsum[m] + Kalpha);
		}
		
		// cummulate multinomial parameters
		for (int k = 1; k < newModel.K; k++){
			newModel.p[k] += newModel.p[k - 1];
		}
		
		// scaled sample because of unnormalized p[]
		double u = Util.random() * newModel.p[newModel.K - 1];
		
		for (topic = 0; topic < newModel.K; topic++){
			if (newModel.p[topic] > u)
				break;
		}
		if (topic >= newModel.K) {
			System.out.println("---error---");
			System.out.println(u);
			System.out.println(newModel.p[topic-1]);
		}
		
		// add newly estimated z_i to count variables
		newModel.nw[w][topic] += 1;
		newModel.nd[m][topic] += 1;
		newModel.nwsum[topic] += 1;
		newModel.ndsum[m] += 1;
		
		return topic;
	}

	@Override
	protected void computeNewTheta() {
		double Kalpha = newModel.K * newModel.alpha;
		for (int m = 0; m < newModel.M; m++){
			for (int k = 0; k < newModel.K; k++){
				newModel.theta[m][k] = (newModel.nd[m][k] + newModel.alpha) / (newModel.ndsum[m] + Kalpha);
			}//end foreach topic
		}//end foreach new document
	}

	@Override
	protected void computeNewPhi() {
		double Vbeta = newModel.V * newModel.beta;
		for (int k = 0; k < newModel.K; k++){
			for (int w = 0; w < newModel.V; w++){
				newModel.phi[k][w] = (newModel.nw[w][k] + bp[w][k] + newModel.beta) / (newModel.nwsum[k] + bpsum[k] + Vbeta);
			}//end foreach word
		}// end foreach topic
	}
	
	@Override
	public void perIterationPostProcess(){
		if (newModel.liter % 100 == 1) {
			System.out.print("iteration " + newModel.liter + ":");
			for (int k=0; k<newModel.K; k++) System.out.print(" " + bpsum[k]); 
			System.out.println();
		}
		for (int k = 0; k < newModel.K; k++){
			double newsum = 0.0;
			for (int w = 0; w < newModel.V; w++){
				bp[w][k] = bp[w][k] * 0.8;
				newsum += bp[w][k];
			}//end foreach word
			bpsum[k] = newsum;
		}// end foreach topic
	}
	
	// prior topics and inferring topics share the same index  
	private void initBetaPrior(double scaling, int[] topics) {
		int V = newModel.V;
		int K = newModel.K;
		bp = new double[V][K];
		bpsum = new double[K];
		
		System.out.print("init beta prior with topics:");
		for (int z : topics) System.out.print(" " + z);
		System.out.println();
		System.out.println("scaling factor = " + scaling);
		System.out.println("beta = " + newModel.beta + ", V = " + newModel.V);
		
		// init beta matrix
		for (int z=0; z<K; z++) {
			bpsum[z] = 0.0;
			for (int w = 0; w < V; w++) {
				bp[w][z] = .0;
			}
		}
		
		// topics with prior
		// TODO local topic share the same topic index with global priors, this requires K >= trnModel.K.
		for (int w = 0; w < V; w++) {
			int gid = newModel.data.local2GlobalWord(w);
			for (int z = 0; z < topics.length; z++) {
				int tz = topics[z];
				bp[w][tz] = trnModel.phi[tz][gid] * scaling;
				bpsum[tz] += bp[w][tz];
			}
		}
		
		// topics without prior
		/*
		boolean[] priorMarks = new boolean[K];
		for (int i=0; i<topics.length; i++) 
			priorMarks[topics[i]] = true; // otherwise false
		for (int z = 0; z < K; z++) {
			if (priorMarks[z]) continue;
			for (int w = 0; w < V; w++) {
				bp[w][z] = newModel.beta;
			}
			bpsum[z] = newModel.beta * V;
		}
		*/
	}

	@Override
	public Model initInfModel(LDADataset newData) {
		newModel = new Model();
		if (!newModel.initParameters(option)) return null;
		assert(newModel.K == trnModel.K);
		
		if (newData == null) {
			newModel.initDataset(option, trnModel.data.dict);
		} else {
			newModel.setDataset(newData);
		}
		
		int NN = 0;
		for (Document d : newModel.data.docs) NN += d.length;
		double scaling = /*newModel.beta * newModel.V **/ 1.0 * NN / newModel.K;
		if (option.cfgFile.compareTo("NONE") == 0 || option.cfgFile.compareTo("ALL") == 0) {
			// local prior
			int[] topics = null;
			if (option.cfgFile.compareTo("NONE") == 0) {
				topics = new int[0];
			} else {
				topics = new int[newModel.K];
				for (int i=0; i<newModel.K; i++) topics[i] = i;
			}
			initBetaPrior(scaling, topics);
			newModel.initCountersByRandom();
			newModel.initZeroProbs();
		}
		else {
			// global prior
			cfg = new TEConfig(trnModel.dir + File.separator + option.cfgFile);
			int slot = option.slot;
			initBetaPrior(scaling, cfg.getPriorTopics(slot));
			newModel.initCountersByZ(trnModel, cfg.getPriorTopics(slot), cfg.getPartition(slot)[0], cfg.getPartition(slot)[1]);
			newModel.initZeroProbs();
		} 
		return newModel;
	}

}
