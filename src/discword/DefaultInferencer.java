package discword;

public class DefaultInferencer extends Inferencer {
	
	public int infSampling(int m, int n) {
		// remove z_i from the count variables
		int topic = newModel.z[m].get(n);
		int _w = newModel.data.docs[m].words[n];
		int w = newModel.data.local2GlobalWord(_w);
		newModel.nw[_w][topic] -= 1;
		newModel.nd[m][topic] -= 1;
		newModel.nwsum[topic] -= 1;
		newModel.ndsum[m] -= 1;
		
		double Vbeta = trnModel.V * newModel.beta;
		double Kalpha = trnModel.K * newModel.alpha;
		
		// do multinomial sampling via cummulative method
		for (int k = 0; k < newModel.K; k++){			
			newModel.p[k] = 
				(trnModel.nw[w][k] + newModel.nw[_w][k] + newModel.beta)/(trnModel.nwsum[k] +  newModel.nwsum[k] + Vbeta) *
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
		
		// add newly estimated z_i to count variables
		newModel.nw[_w][topic] += 1;
		newModel.nd[m][topic] += 1;
		newModel.nwsum[topic] += 1;
		newModel.ndsum[m] += 1;
		
		return topic;
	}
	
	protected void computeNewTheta(){
		for (int m = 0; m < newModel.M; m++){
			for (int k = 0; k < newModel.K; k++){
				newModel.theta[m][k] = (newModel.nd[m][k] + newModel.alpha) / (newModel.ndsum[m] + newModel.K * newModel.alpha);
			}//end foreach topic
		}//end foreach new document
	}
	
	protected void computeNewPhi(){
		for (int k = 0; k < newModel.K; k++){
			for (int _w = 0; _w < newModel.V; _w++){
				Integer id = newModel.data.local2GlobalWord(_w);
				
				if (id != null){
					newModel.phi[k][_w] = (trnModel.nw[id][k] + newModel.nw[_w][k] + newModel.beta) / 
						(trnModel.nwsum[k] + newModel.nwsum[k] + newModel.V * newModel.beta);
				}
			}//end foreach word
		}// end foreach topic
	}

	@Override
	public Model initInfModel(LDADataset newData) {
		Model model = new Model();
		if (!model.initParameters(option)) return null;
		if (newData != null) model.setDataset(newData);
		else if (!model.initDataset(option)) return null;
		
		System.out.println("***ntopics, alpha & beta params are inherited from train model ***");
		model.copyKAlphaBeta(trnModel);
		
		model.initCountersByRandom();
		model.initZeroProbs();
		return model;
	}
}
