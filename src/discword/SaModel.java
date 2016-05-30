package discword;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.Vector;

public class SaModel extends Model{
    //---------------------------------------------------------------
    //  Class Variables
    //---------------------------------------------------------------
    public static String topicSummarySuffix;
    public static String wordSummarySuffix;
    
    //---------------------------------------------------------------
    //  Model Parameters and Variables
    //---------------------------------------------------------------
        
    // Temp variables while sampling
    public Vector<Integer> [] z; //topic assignments for words, size M x doc.size()
    protected double [][] nw; //nw[i][j]: number of instances of word/term i assigned to topic j, size V * K
    protected double [][] nd; //nd[i][j]: number of words in document i assigned to topic j, size M x K
    protected double [] nwsum; //nwsum[j]: total number of words assigned to topic j, size K
    protected double [] ndsum; //ndsum[i]: total number of words in document i, size M
    
    // new parameters
    public int NN;
    public int [] NW;
    public double [][] reversePhi;  // p(z|w) matrix K*V
    public double [] topicProb;     // p(z|Corpus)
    public double [] TK;  // topic knowledge for each word
    
    public boolean dryRun = false;
    
    // temp variables for sampling
    protected double [] p;
    
    //---------------------------------------------------------------
    //  Constructors
    //---------------------------------------------------------------   

    public SaModel(){
        setDefaultValues();
    }
    
    /**
     * Set default values for variables
     */
    public void setDefaultValues(){
        wordMapFile = "wordmap.txt";
        trainlogFile = "trainlog.txt";
        tassignSuffix = ".tassign";
        thetaSuffix = ".theta";
        phiSuffix = ".phi";
        othersSuffix = ".others";
        twordsSuffix = ".twords";
        NDSuffix = ".dz";
        
        topicSummarySuffix = ".topicS";
        wordSummarySuffix = ".wordS";
        
        dir = "./";
        dfile = "trndocs.dat";
        modelName = "model-final";
        modelStatus = Constants.MODEL_STATUS_UNKNOWN;       
        
        M = 0;
        V = 0;
        K = 100;
        alpha = 0.1; // 50 / K
        beta = 0.1;  // 0.1
        niters = 2000;
        liter = 0;
        
        z = null;
        nw = null;      // V * K
        nd = null;      // M * K
        nwsum = null;   // K
        ndsum = null;   // M
        theta = null;
        phi = null;
    }
    
    //---------------------------------------------------------------
    //  I/O Methods
    //---------------------------------------------------------------
    /**
     * read other file to get parameters
     */
    protected boolean readOthersFile(String otherFile){
        //open file <model>.others to read:
        
        try {
            BufferedReader reader = new BufferedReader(new FileReader(otherFile));
            String line;
            while((line = reader.readLine()) != null){
                StringTokenizer tknr = new StringTokenizer(line,"= \t\r\n");
                
                int count = tknr.countTokens();
                if (count != 2)
                    continue;
                
                String optstr = tknr.nextToken();
                String optval = tknr.nextToken();
                
                if (optstr.equalsIgnoreCase("alpha")){
                    alpha = Double.parseDouble(optval);                 
                }
                else if (optstr.equalsIgnoreCase("beta")){
                    beta = Double.parseDouble(optval);
                }
                else if (optstr.equalsIgnoreCase("ntopics")){
                    K = Integer.parseInt(optval);
                }
                else if (optstr.equalsIgnoreCase("liter")){
                    liter = Integer.parseInt(optval);
                }
                else if (optstr.equalsIgnoreCase("nwords")){
                    V = Integer.parseInt(optval);
                }
                else if (optstr.equalsIgnoreCase("ndocs")){
                    M = Integer.parseInt(optval);
                }
                else if (optstr.equalsIgnoreCase("perplexity")){
                    // ignore
                }
                else {
                    // any more?
                }
            }
            
            reader.close();
        }
        catch (Exception e){
            System.out.println("Error while reading other file:" + e.getMessage());
            e.printStackTrace();
            return false;
        }
        return true;
    }
    
    @SuppressWarnings("unchecked")
    protected boolean readTAssignFile(String tassignFile){
        try {
            int i,j;
            BufferedReader reader = new BufferedReader(new InputStreamReader(
                    new FileInputStream(tassignFile), "UTF-8"));
            
            String line;
            z = new Vector[M];          
            data = new LDADataset(M);
            data.V = V;         
            for (i = 0; i < M; i++){
                line = reader.readLine();
                StringTokenizer tknr = new StringTokenizer(line, " \t\r\n");
                
                int length = tknr.countTokens();
                
                Vector<Integer> words = new Vector<Integer>();
                Vector<Integer> topics = new Vector<Integer>();
                
                for (j = 0; j < length; j++){
                    String token = tknr.nextToken();
                    
                    StringTokenizer tknr2 = new StringTokenizer(token, ":");
                    if (tknr2.countTokens() != 2){
                        System.out.println("Invalid word-topic assignment line\n");
                        return false;
                    }
                    
                    words.add(Integer.parseInt(tknr2.nextToken()));
                    topics.add(Integer.parseInt(tknr2.nextToken()));
                }//end for each topic assignment
                
                //allocate and add new document to the corpus
                Document doc = new Document(words);
                data.setDoc(doc, i);
                
                //assign values for z
                z[i] = new Vector<Integer>();
                for (j = 0; j < topics.size(); j++){
                    z[i].add(topics.get(j));
                }
                
            }//end for each doc
            
            reader.close();
        }
        catch (Exception e){
            System.out.println("Error while loading model: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
        return true;
    }
    
    /**
     * load saved model
     */
    public boolean loadModel(){
        if (!readOthersFile(dir + File.separator + modelName + othersSuffix))
            return false;
        
        if (!readTAssignFile(dir + File.separator + modelName + tassignSuffix))
            return false;
        
        // read dictionary
        Dictionary dict = new Dictionary();
        if (!dict.readWordMap(dir + File.separator + modelName + "." + wordMapFile))
            return false;
            
        data.dict = dict;
        
        return true;
    }
    
    /**
     * Save word-topic assignments for this model
     */
    public boolean saveModelTAssign(String filename){
        int i, j;
        
        try{
            BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
            
            //write docs with topic assignments for words
            for (i = 0; i < data.M; i++){
                for (j = 0; j < data.docs[i].length; ++j){
                    writer.write(data.docs[i].words[j] + ":" + z[i].get(j) + " ");                  
                }
                writer.write("\n");
            }
                
            writer.close();
        }
        catch (Exception e){
            System.out.println("Error while saving model tassign: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
        return true;
    }
    
    public boolean saveModelND(String filename) {
        int i, j;
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
            for (i=0; i<data.M; i++) {
                for (j=0; j<K; j++){
                    writer.write( (j==0 ? "" : "\t") + nd[i][j]);
                }
                writer.write("\n");
            }
        } catch (Exception e) {
            System.out.println("Error while saving model ND counters: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
        return true;
    }
    
    public boolean saveModelND() {
        String filename = dir + File.separator + modelName + NDSuffix;
        return saveModelND(filename);
    }
    
    /*
     * Yu Jiang
     */
    public double perplexity_dry() {
        double val = 0.0;
        
        val += Gamma.logGamma(alpha * K) * M;
        val -= Gamma.logGamma(alpha) * K * M;
        
        for (int d=0; d<M; d++) {
            double [] tmp_nd = new double[K];
            for (int i=0; i<data.docs[d].length; i++) {
                int w = data.docs[d].words[i];
                int topic = z[d].get(i);
                
                tmp_nd[topic] += TK[w];
            }
            
            for (int j = 0; j < K; j++) {
                val += Gamma.logGamma(alpha + tmp_nd[j]);
            }
            val -= Gamma.logGamma(alpha * K + Util.sum(tmp_nd));
        }
        
        val += Gamma.logGamma(beta * V) * K;
        val -= Gamma.logGamma(beta) * V * K;
        
        double overall_sum = 0.0;
        for (int i = 0; i < K; i++) {
            double tmp_nw_sum = 0.0;
            for (int j = 0; j < V; j++) {
                tmp_nw_sum += nw[j][i] * TK[j];
                val += Gamma.logGamma(beta + nw[j][i] * TK[j]);
            }
            overall_sum += tmp_nw_sum;
            val -= Gamma.logGamma(beta * V + tmp_nw_sum);
        }
        
        double N = overall_sum;
        val = Math.exp(- val / N);
        return val;
    }
    
    public double perplexity() {
        return perplexity(nd, nw, ndsum, nwsum);
    }
    
    public double perplexity(double[][] nd, double[][] nw, double[] ndsum, double[] nwsum) {
        double val = 0.0;
        
        val += Gamma.logGamma(alpha * K) * M;
        val -= Gamma.logGamma(alpha) * K * M;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                val += Gamma.logGamma(alpha + nd[i][j]);
            }
            val -= Gamma.logGamma(alpha * K + ndsum[i]);
        }
        
        val += Gamma.logGamma(beta * V) * K;
        val -= Gamma.logGamma(beta) * V * K;
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < V; j++) {
                val += Gamma.logGamma(beta + nw[j][i]);
            }
            val -= Gamma.logGamma(beta * V + nwsum[i]);
        }
        
        val = Math.exp(- val / Util.sum(ndsum));
        return val;
    }
    
    /**
     * Save theta (topic distribution) for this model
     */
    public boolean saveModelTheta(String filename){
        try{
            BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
            for (int i = 0; i < M; i++){
                for (int j = 0; j < K; j++){
                    writer.write(theta[i][j] + " ");
                }
                writer.write("\n");
            }
            writer.close();
        }
        catch (Exception e){
            System.out.println("Error while saving topic distribution file for this model: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
        return true;
    }
    
    /**
     * Save word-topic distribution
     */
    
    public boolean saveModelPhi(String filename){
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
            
            for (int i = 0; i < K; i++){
                for (int j = 0; j < V; j++){
                    writer.write(phi[i][j] + " ");
                }
                writer.write("\n");
            }
            writer.close();
        }
        catch (Exception e){
            System.out.println("Error while saving word-topic distribution:" + e.getMessage());
            e.printStackTrace();
            return false;
        }
        return true;
    }
    
    /**
     * Save other information of this model
     */
    public boolean saveModelOthers(String filename){
        try{
            BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
            
            writer.write("alpha=" + alpha + "\n");
            writer.write("beta=" + beta + "\n");
            writer.write("ntopics=" + K + "\n");
            writer.write("ndocs=" + M + "\n");
            writer.write("nwords=" + V + "\n");
            writer.write("liters=" + liter + "\n");
            writer.write("perplexity=" + perplexity() + "\n");
            writer.write("complexity=" + complexity() + "\n");
            for (Map.Entry entry : avgWordSparsity().entrySet()) {
                writer.write("word sparsity " + entry.getKey() + " = " + entry.getValue() + "\n");
            }
            for (Map.Entry entry : avgTopicSparsity().entrySet()) {
                writer.write("topic sparsity " + entry.getKey() + " = " + entry.getValue() + "\n");
            }
            for (Map.Entry entry : avgTopicDistance().entrySet()) {
                writer.write("topic distance " + entry.getKey() + " = " + entry.getValue() + "\n");
            }
            writer.write("dryrun=" + dryRun + "\n");
            
            writer.close();
        }
        catch(Exception e){
            System.out.println("Error while saving model others:" + e.getMessage());
            e.printStackTrace();
            return false;
        }
        return true;
    }
    
    /**
     * Save model the most likely words for each topic
     */
    public boolean saveModelTwords(String filename){
        try{
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(filename), "UTF-8"));
            
            if (twords > V){
                twords = V;
            }
            
            for (int k = 0; k < K; k++){
                List<Pair> wordsProbsList = new ArrayList<Pair>(); 
                for (int w = 0; w < V; w++){
                    Pair p = new Pair(w, phi[k][w], false);
                    
                    wordsProbsList.add(p);
                }//end foreach word
                
                //print topic               
                writer.write("Topic " + k + "th:\n");
                Collections.sort(wordsProbsList);
                
                for (int i = 0; i < twords; i++){
                    if (data.dict.contains((Integer)wordsProbsList.get(i).first)){
                        String word = data.dict.getWord((Integer)wordsProbsList.get(i).first);
                        
                        writer.write("\t" + word + " " + wordsProbsList.get(i).second + "\n");
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
    
    /**
     * Save model the most likely words with weights for each topic.
     */
    public boolean saveModelTopicSummary(String filename){
        try{
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(filename), "UTF-8"));
            
            if (twords > V){
                twords = V;
            }
            List<Pair> wordsProbsList = new ArrayList<Pair>(V);
            List<Pair> wordsWeightsList = new ArrayList<Pair>(V);
            
            List<Integer> powIndex = new ArrayList<Integer>();
            int jj = 1;
            do {
                powIndex.add(jj);
                jj *= 2;
            } while (jj <= V);
            if (jj < 2*V) powIndex.add(V);
            double[][] pct = new double[256][K];
            
            for (int k = 0; k < K; k++){
                wordsProbsList.clear();
                wordsWeightsList.clear();
                
                //print topic
                writer.write(String.format("Topic %dth - probability=%.6f:\n", k, topicProb[k]));
                
                // top words of probability 
                for (int w = 0; w < V; w++){
                    Pair p = new Pair(w, phi[k][w], false);
                    wordsProbsList.add(p);
                }//end foreach word
                Collections.sort(wordsProbsList);
                
                // top words of entropy based weights prob * deterministic
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
                
                // compute topic percentiles
                double pct_value = 0.0;
                int next = 0;
                for (int i=1; i<=256; i++) {
                    pct_value += (Double)(wordsProbsList.get(i-1).second);
//                  if (i == powIndex.get(next)) {
//                      pct[next++][k] = pct_value;
//                  }
                    pct[i-1][k] = pct_value;
                }
            } //end foreach topic
            
            // compute percentiles statistics
            writer.write("\ntopic percentiles distribution:\n");
            for (jj=0; jj<256; jj++) {
                Pair pair = Util.AvgStdDev(pct[jj]);
                writer.write(String.format("%d %d %.3f %.3f\n", jj, jj, pair.first, pair.second));
            }
            
            // Inter-topic distance
                        
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
                writer.write(String.format("%16s %6d %6f %.6f %.6f\n", word, NW[w], TK[w], wordList[i].second, Util.infoRate(nw[w])));
            }
            writer.write("\n\n\n\n\n");
            
            for (int w = 0; w < V; w++){
                wordList[w] = new Pair(w, TK[w], true);
            }
            Arrays.sort(wordList);
            for (int i=0; i<V/10; i++) {
                int w = (Integer)wordList[i].first;
                String word = data.dict.getWord(w);
                writer.write(String.format("%16s %6d %6f %.6f %.6f\n", word, NW[w], TK[w], TK[w] * NW[w], Util.infoRate(nw[w])));
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
    
    /**
     * Save model
     */
    public boolean saveModel(String modelName){
        if (!saveModelTAssign(dir + File.separator + modelName + tassignSuffix)){
            return false;
        }
        
        if (!saveModelOthers(dir + File.separator + modelName + othersSuffix)){
            return false;
        }
        
        if (!saveModelTheta(dir + File.separator + modelName + thetaSuffix)){
            return false;
        }
        
        if (!saveModelPhi(dir + File.separator + modelName + phiSuffix)){
            return false;
        }
        
        if (twords > 0){
            if (!saveModelTwords(dir + File.separator + modelName + twordsSuffix))
                return false;
            if (!saveModelWordSummary(dir + File.separator + modelName + wordSummarySuffix))
                return false;
            if (!saveModelTopicSummary(dir + File.separator + modelName + topicSummarySuffix))
                return false;
            
            // should be the last step, as NW and ND are updated.
            hackOriginalPhi();
            if (!saveModelTopicSummary(dir + File.separator + modelName + topicSummarySuffix + "_original"))
                return false;
        }
        return true;
    }
    
    //---------------------------------------------------------------
    //  Init Methods
    //---------------------------------------------------------------
    /**
     * initialize the model
     */
    protected boolean initParameters(LDACmdOption option){
        if (option == null)
            return false;
        
        modelName = option.modelName;
        K = option.K;
        p = new double[K];
        
        if (option.alpha > 0.0)
            alpha = option.alpha;
        
        if (option.beta > 0.0)
            beta = option.beta;
        
        niters = option.niters;
        
        dir = option.dir;
        if (dir.endsWith(File.separator))
            dir = dir.substring(0, dir.length() - 1);
        
        dfile = option.dfile;
        twords = option.twords;
        wordMapFile = option.wordMapFileName;
        savestep = option.savestep;
        
        dryRun = option.entropyDryRun;
        return true;
    }
    
    public void setDataset(LDADataset newData){
        data = newData;
        M = data.M;
        V = data.V;
    }
    
    protected boolean initDataset(LDACmdOption option) {
        if (option.useDict) {
            String filepath = option.dir + File.separator + option.wordMapFileName;
            System.out.println("use existing wordmap: " + filepath);
            Dictionary dict = new Dictionary();
            dict.readWordMap(filepath);
            return initDataset(option, dict);
        }
        return initDataset(option, null);
    }
    
    protected boolean initDataset(LDACmdOption option, Dictionary globalDict) {
        data = LDADataset.readDataSet(dir + File.separator + dfile, globalDict);
        if (data == null){
            System.out.println("Fail to read training data!\n");
            return false;
        }
        
        //+ allocate memory and assign values for variables     
        M = data.M;
        V = data.V;
        
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
        theta = new double[M][K];       
        phi = new double[K][V];
        reversePhi = new double[K][V];
        topicProb = new double[K];
        
        TK = new double[V];
        Arrays.fill(TK, 1.0);
        return true;
    }
    
    public boolean copyKAlphaBeta(Model trnModel) {
        K = trnModel.K;
        alpha = trnModel.alpha;
        beta = trnModel.beta;
        p = new double[K];
        return true;
    }
    
    public boolean initEstModel(LDACmdOption option) {
        if (initParameters(option) &&
                initDataset(option) &&
                initCountersByRandom() &&
                initZeroProbs())
            return true;
        return false;
    }
    
    public boolean initInfModel(LDACmdOption option, Model trnModel) {
        System.out.println("***ntopics, alpha & beta params are inherited from train model ***");
        if (!initDataset(option)) return false;
        return initInfModel(option, data, trnModel);
    }
    
    public boolean initInfModel(LDACmdOption option, LDADataset newData, Model trnModel) {
        System.out.println("***ntopics, alpha & beta params are inherited from train model ***");
        if (initParameters(option)) {
            copyKAlphaBeta(trnModel);
            
            data = newData;
            initCountersByRandom();
            initZeroProbs();
            return true;
        }
        return false;
    }
    
    public boolean initZeroCounters() {
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
    
    @SuppressWarnings("unchecked")
    public boolean initCountersByZ(Model trnModel, int[] prior_topics, int startDoc, int endDoc){
        initZeroCounters();
        int m,n,k;
        assert (M == endDoc - startDoc);
        z = new Vector[M];
        
        HashSet<Integer> zset = new HashSet<Integer>();
        for (int topic : prior_topics){
            zset.add(topic);
        }

        ArrayList<Integer> zothers = new ArrayList<Integer>();
        for(int topic=0; topic<K; topic++){
            if (!zset.contains(topic))
                zothers.add(topic);
        }
        int KK = zothers.size();
        
        for (m=0; m<M; m++){
            int N = data.docs[m].length;
            assert (N == trnModel.data.docs[startDoc + m].length);
            z[m] = new Vector<Integer>();
            for (n=0; n<N; n++){
                k = trnModel.z[m + startDoc].get(n);
                if (!zset.contains(k)){
                    k = zothers.get((int)Math.floor(Util.random() * KK));
                }
                z[m].add(k);
                
                nw[data.docs[m].words[n]][k] += 1;
                nd[m][k] += 1;
                nwsum[k] += 1;
            }
            ndsum[m] = N;
        }
        return true;
    }
    
    @SuppressWarnings("unchecked")
    public boolean initCountersByBetaPrior(Model trnModel) {
        initZeroCounters();
        
        int m,n,k;
        z = new Vector[M];
        
        for (m = 0; m < data.M; m++){
            int N = data.docs[m].length;
            z[m] = new Vector<Integer>();
            
            //initilize for z
            for (n = 0; n < N; n++){
                int _w = this.data.docs[m].words[n];
                int w = this.data.local2GlobalWord(_w);
                int topic;
                
                for (k=0; k<this.K; k++) {
                    this.p[k] = trnModel.phi[k][w];
                }
                for (k=1; k<this.K; k++) {
                    this.p[k] += this.p[k-1];
                }
                
                double u = Util.random() * this.p[this.K - 1];
                for (topic = 0; topic < this.K; topic++){
                    if (this.p[topic] > u)
                        break;
                }
                
                z[m].add(topic);
                
                // number of instances of word assigned to topic j
                nw[data.docs[m].words[n]][topic] += 1;
                // number of words in document i assigned to topic j
                nd[m][topic] += 1;
                // total number of words assigned to topic j
                nwsum[topic] += 1;
            }
            // total number of words in document i
            ndsum[m] = N;
        }
        return true;
    }
    
    @SuppressWarnings("unchecked")
    public boolean initCountersByRandom() {
        initZeroCounters();
        
        int m,n;
        z = new Vector[M];
        
        for (m = 0; m < data.M; m++){
            int N = data.docs[m].length;
            z[m] = new Vector<Integer>();
            
            //initialize for z
            for (n = 0; n < N; n++){
                int topic = (int)Math.floor(Util.random() * K);
                z[m].add(topic);
                
                // number of instances of word assigned to topic j
                nw[data.docs[m].words[n]][topic] += 1;
                // number of words in document i assigned to topic j
                nd[m][topic] += 1;
                // total number of words assigned to topic j
                nwsum[topic] += 1;
            }
            // total number of words in document i
            ndsum[m] = N;
        }
        return true;
    }
        
    /**
     * init parameter for continue estimating or for later inference
     */
    public boolean initEstcModel(LDACmdOption option){
        if (!initParameters(option))
            return false;
        
        int m, n, w;
        
        // load model, i.e., read z and trndata
        if (!loadModel()){
            System.out.println("Fail to load word-topic assignment file of the model!\n");
            return false;
        }
        
        System.out.println("Model loaded:");
        System.out.println("\talpha:" + alpha);
        System.out.println("\tbeta:" + beta);
        System.out.println("\tM:" + M);
        System.out.println("\tV:" + V);
        System.out.println("\tK:" + K);
        p = new double[K];
        
        initZeroCounters();
        
        for (m = 0; m < data.M; m++){
            int N = data.docs[m].length;
            
            // assign values for nw, nd, nwsum, and ndsum
            for (n = 0; n < N; n++){
                w = data.docs[m].words[n];
                int topic = (Integer)z[m].get(n);
                
                // number of instances of word i assigned to topic j
                nw[w][topic] += 1;
                // number of words in document i assigned to topic j
                nd[m][topic] += 1;
                // total number of words assigned to topic j
                nwsum[topic] += 1;  
            }
            // total number of words in document i
            ndsum[m] = N;
        }
        
        initZeroProbs();
        
        return true;
    }
    
    public void resetCounters() {
        int w, m, k;
        for (w = 0; w < V; w++){
            for (k = 0; k < K; k++){
                nw[w][k] = 0;
            }
        }
        
        for (m = 0; m < M; m++){
            for (k = 0; k < K; k++){
                nd[m][k] = 0;
            }
        }
        
        for (k = 0; k < K; k++){
            nwsum[k] = 0;
        }
        
        for (m = 0; m < M; m++){
            ndsum[m] = 0;
        }
    }
    
    public void computeReversePhi() {
        for (int w=0; w<V; w++) {
            double total = Util.sum(nw[w]) + betaSum();
            for (int k=0; k<K; k++) {
                reversePhi[k][w] = (nw[w][k] +beta(w)) / total;
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
        computeWordInfoRate(TK);
    }
    
    public void computeWordInfoRate(double[] TK) {
        if (dryRun) return;  // does nothing TK[w] = 1.0 for any w.
        
        for (int w = 0; w < V; w++) {
            double total = Util.sum(nw[w]) + betaSum();
            for (int k = 0; k < K; k++) {
                assert nw[w][k] >= 0.0 : String.format("nw < 0 - %d-%d-%f", w,
                        k, nw[w][k]);
                p[k] = (nw[w][k] + beta(w)) / total;
            }
            TK[w] = Math.max(TK[w] - 0.1, Util.infoRate(p));
        }
    }
    
    public double betaSum() {
        return beta * K;
    }
    public double beta(int w) {
        return beta;
    }
    private double weightedWordInfoRate(double infoRate, double factor) {
        if (factor == 0.0) return 1.0;
        else if (factor == 1.0) return infoRate;
        else return Math.pow(infoRate, factor);
    }
    
    public HashMap<String, Double> avgWordSparsity() {
        double [] tmp = new double[V];
        double x=0, y=0, z=0;
        for (int k=0; k<K; k++) {
            for (int w=0; w<V; w++) tmp[w] = nw[w][k];
            x += Util.HoyerSparsity(tmp);
            y += Util.infoRate(tmp);
            z += Util.GiniSparsity(tmp);
        }
        HashMap<String, Double> map = new HashMap<String, Double>();
        map.put("hoyer", x / K);
        map.put("information-ratio", y / K);
        map.put("gini", z / K);
        return map;
    }
    
    public HashMap<String, Double> avgTopicSparsity() {
        double x=0, y=0, z=0;
        for (int d=0; d<M; d++) {
            x += Util.HoyerSparsity(nd[d]);
            y += Util.infoRate(nd[d]);
            z += Util.GiniSparsity(nd[d]);
        }
        HashMap<String, Double> map = new HashMap<String, Double>();
        map.put("hoyer", x / M);
        map.put("information-ratio", y / M);
        map.put("gini", z / M);
        return map;
    }
    
    public HashMap<String, Double> avgTopicDistance() {
        double total = 0.0;
        int count = 0;
        double[] arr = new double[K*(K-1)];
        for (int k=0; k<K; k++) {
            for (int j=0; j<k; j++) {
                double x= Util.KLDivergence(phi[k], phi[j]);
                total += x;
                arr[count] = x;
                count ++;
            }
        }
        HashMap<String, Double> map = new HashMap<String, Double>();
        map.put("kl", total / count);
        map.put("kl-stddev", (Double)(Util.AvgStdDev(arr).second));
        System.out.println("count = " + count);
        return map;
    }
    
    // wang, c., blei, D. Decoupling sparsity and smoothness in the discrete hierarchical Dirichlet process.
    public int complexity() {
        int complexity = 0;
        for (int m=0; m<M; m++){
            Set<Integer> unique_words = new HashSet<Integer>();
            int N = data.docs[m].length;
            for (int n=0; n<N; n++){
                int k = z[m].get(n);
                int w = data.docs[m].words[n];
                unique_words.add(k*V + w);
            }
            complexity += unique_words.size();
        }
        return complexity;
    }
    
    public void hackOriginalPhi() {
        initZeroCounters();
        for (int m=0; m<M; m++){
            int N = data.docs[m].length;
            for (int n=0; n<N; n++){
                int k = z[m].get(n);
                int w = data.docs[m].words[n];
                nw[w][k] += 1;
                nwsum[k] += 1;
                nd[m][k] += 1;
                ndsum[m] += 1;
            }
        }
        // compute theta
        for (int m = 0; m < M; m++){
            for (int k = 0; k < K; k++){
                theta[m][k] = (nd[m][k] + alpha) / (ndsum[m] + K * alpha);
            }
        }
        // compute phi
        for (int k = 0; k < K; k++){
            for (int w = 0; w < V; w++){
                phi[k][w] = (nw[w][k] + beta) / (nwsum[k] + V * beta);
            }
        }
        computeReversePhi();
    }
}
