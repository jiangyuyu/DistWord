package discword;

public class Assigner {
	public Model trnModel;
	public Model newModel;
	private LDACmdOption option;
	
	
	public boolean init(LDACmdOption option){
		this.option = option;
		
		trnModel = new Model();
		LDACmdOption trnOption = new LDACmdOption();
		// use the same dir & dfile for train model by if empty
		trnOption.dir = (option.trnDir.length() == 0 ? option.dir : option.trnDir);
		trnOption.dfile = (option.trnDFile.length() == 0 ? option.dfile : option.trnDFile);
		trnOption.modelName = option.trnModelName;
		trnOption.wordMapFileName = option.trnWordMapFileName;
		if (!trnModel.initEstcModel(trnOption))
			return false;		
		computeTheta(trnModel);
		computePhi(trnModel);
		
		Model newModel = new Model();
		newModel.initInfModel(option, trnModel.data, trnModel);
		newModel.resetCounters();
		this.newModel = newModel;
		
		return true;
	}
	
	public Model assign() {
		System.out.println("Sampling " + option.niters + " iteration for assignment!");
		
		for (trnModel.liter = 1; trnModel.liter <= option.niters; trnModel.liter++){
			//System.out.println("Iteration " + newModel.liter + " ...");
			
			// for all newz_i
			for (int m = 0; m < trnModel.M; ++m){
				for (int n = 0; n < trnModel.data.docs[m].length; n++){
					// (newz_i = newz[m][n]
					// sample from p(z_i|z_-1,w)
					int topic = assSampling(m, n);
					int w = trnModel.data.docs[m].words[n];
					newModel.nw[w][topic] += 1;
					newModel.nd[m][topic] += 1;
					newModel.nwsum[topic] += 1;
					newModel.ndsum[m] += 1;
				}
			}//end foreach new doc
			
		}// end iterations
		
		System.out.println("Gibbs sampling for assignment completed!");
		
		computeTheta(newModel);
		computePhi(newModel);
		
		System.out.println("Saving the assignment outputs!");
		newModel.saveModelND();
		return newModel;
	}
	
	protected int assSampling(int m, int n){
		// remove z_i from the count variables
		int topic = trnModel.z[m].get(n);
		int w = trnModel.data.docs[m].words[n];
				
		// do multinomial sampling via cummulative method		
		for (int k = 0; k < trnModel.K; k++){		
			trnModel.p[k] = trnModel.theta[m][k] * trnModel.phi[k][w];
		}
		
		// cummulate multinomial parameters
		for (int k = 1; k < trnModel.K; k++){
			trnModel.p[k] += trnModel.p[k - 1];
		}
		
		// scaled sample because of unnormalized p[]
		double u = Util.random() * trnModel.p[trnModel.K - 1];
		
		for (topic = 0; topic < trnModel.K; topic++){
			if (trnModel.p[topic] > u)
				break;
		}
				
		return topic;
	}
	
	protected void computeTheta(Model model){
		for (int m = 0; m < model.M; m++){
			for (int k = 0; k < model.K; k++){
				model.theta[m][k] = (model.nd[m][k] + model.alpha) / (model.ndsum[m] + model.K * model.alpha);
			}
		}
	}
	
	protected void computePhi(Model model){
		for (int k = 0; k < model.K; k++){
			for (int w = 0; w < model.V; w++){
				model.phi[k][w] = (model.nw[w][k] + model.beta) / (model.nwsum[k] + model.V * model.beta);
			}
		}
	}
}
