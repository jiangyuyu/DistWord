/*
 * Copyright (C) 2007 by
 * 
 * 	Xuan-Hieu Phan
 *	hieuxuan@ecei.tohoku.ac.jp or pxhieu@gmail.com
 * 	Graduate School of Information Sciences
 * 	Tohoku University
 * 
 *  Cam-Tu Nguyen
 *  ncamtu@gmail.com
 *  College of Technology
 *  Vietnam National University, Hanoi
 *
 * JGibbsLDA is a free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 *
 * JGibbsLDA is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with JGibbsLDA; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
 */

package discword;

import java.io.File;


public abstract class Inferencer {	
	// Train model
	public Model trnModel;
	public Dictionary globalDict;
	protected LDACmdOption option;
	
	protected Model newModel;
	
	//-----------------------------------------------------
	// Init method
	//-----------------------------------------------------
	public boolean init(LDACmdOption option){
		this.option = option;
		
		// load train model
		trnModel = new Model();
		LDACmdOption trnOption = new LDACmdOption();
		trnOption.dir = (option.trnDir.length() == 0 ? option.dir : option.trnDir);
		trnOption.dfile = (option.trnDFile.length() == 0 ? option.dfile : option.trnDFile);
		trnOption.modelName = option.trnModelName;
		trnOption.wordMapFileName = option.trnWordMapFileName;
		if (!trnModel.initEstcModel(trnOption))
			return false;
		computeTrnTheta();
		computeTrnPhi();
		
		globalDict = trnModel.data.dict;
		return true;
	}
	
	public Model inference(String filename) {
		LDADataset dst = LDADataset.readDataSet(filename, globalDict);
		return inference(dst);
	}
	
	public Model inference(String [] strs){		
		LDADataset dataset = LDADataset.readDataSet(strs, globalDict);
		return inference(dataset);
	}
	
	//inference new model ~ getting data from a specified dataset
	public Model inference(LDADataset newData){
		return innerInference(newData);
	}
	
	//inference new model ~ getting dataset from file specified in option
	public Model inference(){
		return innerInference(null);
	}
	
	public abstract Model initInfModel(LDADataset newData);

	/**
	 * do sampling for inference
	 * m: document number
	 * n: word number?
	 */
	protected abstract int infSampling(int m, int n);
	
	protected abstract void computeNewTheta();
	
	protected abstract void computeNewPhi();
	
	public Model innerInference(LDADataset newData) {
		newModel = initInfModel(newData);
		
		System.out.println("Sampling " + option.niters + " iteration for inference!");
		for (newModel.liter = 1; newModel.liter <= option.niters; newModel.liter++){
			//System.out.println("Iteration " + newModel.liter + " ...");
			
			// for all newz_i
			for (int m = 0; m < newModel.M; ++m){
				for (int n = 0; n < newModel.data.docs[m].length; n++){
					// (newz_i = newz[m][n]
					// sample from p(z_i|z_-1,w)
					int topic = infSampling(m, n);
					newModel.z[m].set(n, topic);
				}
			}//end foreach new doc
			perIterationPostProcess();
		}// end iterations
		
		System.out.println("Gibbs sampling for inference completed!");		
		System.out.println("Saving the inference outputs!");
		
		computeNewTheta();
		computeNewPhi();
		newModel.saveModel(option.modelName);
		newModel.data.dict.writeWordMap(option.dir + File.separator + option.modelName + "." + option.wordMapFileName);
		
		return newModel;
	}
	
	public void perIterationPostProcess(){
		//do nothing
	}
			
	protected void computeTrnTheta(){
		for (int m = 0; m < trnModel.M; m++){
			for (int k = 0; k < trnModel.K; k++){
				trnModel.theta[m][k] = (trnModel.nd[m][k] + trnModel.alpha) / (trnModel.ndsum[m] + trnModel.K * trnModel.alpha);
			}
		}
	}
	
	protected void computeTrnPhi(){
		for (int k = 0; k < trnModel.K; k++){
			for (int w = 0; w < trnModel.V; w++){
				trnModel.phi[k][w] = (trnModel.nw[w][k] + trnModel.beta) / (trnModel.nwsum[k] + trnModel.V * trnModel.beta);
			}
		}
	}
}
