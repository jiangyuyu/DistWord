package discword;

import org.kohsuke.args4j.*;

public class LDACmdOption {
	
	@Option(name="-est", usage="Specify whether we want to estimate model from scratch")
	public boolean est = false;
	
	@Option(name="-estw", usage="Specify whether we want to estimate weighted model from scratch")
	public boolean estw = false;
	
	@Option(name="-estw2", usage="Specify whether we want to estimate weighted-2 model from scratch")
	public boolean estw2 = false;
	
	@Option(name="-estc", usage="Specify whether we want to continue the last estimation")
	public boolean estc = false;
	
	@Option(name="-inf", usage="Specify whether we want to do inference")
	public boolean inf = false;
	
	@Option(name="-ass", usage="Specify whether we want to do topic assignment")
	public boolean ass = false;

	@Option(name="-dir", usage="Specify directory")
	public String dir = "";
	
	@Option(name="-dfile", usage="Specify data file")
	public String dfile = "";
	
	@Option(name="-model", usage="Specify the model name")
	public String modelName = "model-final";
	
	@Option(name="-alpha", usage="Specify alpha")
	public double alpha = -1.0;
	
	@Option(name="-beta", usage="Specify beta")
	public double beta = -1.0;
	
	@Option(name="-ntopics", usage="Specify the number of topics")
	public int K = 10;
	
	@Option(name="-niters", usage="Specify the number of iterations")
	public int niters = 1000;
	
	@Option(name="-savestep", usage="Specify the number of steps to save the model since the last save")
	public int savestep = 100;
	
	@Option(name="-twords", usage="Specify the number of most likely words to be printed for each topic")
	public int twords = 20;
	
	@Option(name="-withrawdata", usage="Specify whether we include raw data in the input")
	public boolean withrawdata = false;
	
	@Option(name="-wordmap", usage="Specify the wordmap file to be written")
	public String wordMapFileName = "wordmap.txt";

	// train model setting
	@Option(name="-trndir", usage="Specify the train model's dir")
	public String trnDir = "";
	@Option(name="-trndfile", usage="Specify train model's data file")
	public String trnDFile = "";
	@Option(name="-trnmodel", usage="Specify the train model name")
	public String trnModelName = "model-final";
	@Option(name="-trnwordmap", usage="Specify the wordmap file to be written")
	public String trnWordMapFileName = "wordmap.txt";
	@Option(name="-cfgfile", usage="topic evolution config file.")
	public String cfgFile = "topic_prior.json";
	@Option(name="-slot", usage="time slot index in partition.")
	public int slot = -1;

	// action
	@Option(name="-job", usage="Specify the action to be performed.")
	public String job = "";
	
	@Option(name="-dryrun", usage="entropy model dry run.")
	public boolean entropyDryRun = false;
	
	@Option(name="-usedict", usage="use provided word map")
	public boolean useDict = false;
	
	public JobType jobType = JobType.UNKNOWN;
}
