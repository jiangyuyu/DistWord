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

import org.kohsuke.args4j.*;

public class LDA {
	
	public static void main(String args[]){
		LDACmdOption option = new LDACmdOption();
		CmdLineParser parser = new CmdLineParser(option);
		
		try {
			if (args.length == 0){
				showHelp(parser);
				return;
			}
			
			parser.parseArgument(args);
			
			JobType jt;
			if (option.job.length() > 0) {
				jt = JobType.valueOf(option.job);
			} else {
				if (option.est) jt = JobType.EST;
				else if (option.estc) jt = JobType.ESTC;
				else if (option.estw) jt = JobType.ESTW;
				else if (option.estw2) jt = JobType.ESTW2;
				else if (option.inf) jt = JobType.DefaultINF;
				else if (option.ass) jt = JobType.ASS;
				else {
					throw new IllegalArgumentException("no job is given"); 
				}
			}
			option.jobType = jt;
			
			switch (jt) {
			case EST:
			case ESTC:
				Estimator estimator = new Estimator();
				if (estimator.init(option))
					estimator.estimate();
				break;
			case ESTW:
				//estimator = new TopicDiscriminativeEstimator();
			    estimator = new SaEstimator();
				if (estimator.init(option))
					estimator.estimate();
				break;
			case ESTW2:
				estimator = new W2Estimator();
				if (estimator.init(option))
					estimator.estimate();
				break;
			case DefaultINF:
				Inferencer inferencer = new DefaultInferencer();
				inferencer.init(option);
				inferencer.inference();
				break;
			case BetaPriorINF:
				inferencer = new BetaPriorInferencer();
				inferencer.init(option);
				inferencer.inference();
				break;
			case ASS:
				Assigner ass = new Assigner();
				ass.init(option);
				ass.assign();
				break;
			default:
				System.out.println("No action is specified.");
			}
		}
		catch (CmdLineException cle){
			System.out.println("Command line error: " + cle.getMessage());
			showHelp(parser);
			return;
		}
		catch (IllegalArgumentException iae) {
			System.out.println("unsupported job type error: " + option.job);
			String help = "supported job types:";
			for (JobType type : JobType.values()) {
				help += " " + type.name();
			}
			System.out.println(help);
			return;
		}
		catch (Exception e){
			System.out.println("Error in main: " + e.getMessage());
			e.printStackTrace();
			return;
		}
	}
	
	public static void showHelp(CmdLineParser parser){
		System.out.println("LDA [options ...] [arguments...]");
		parser.printUsage(System.out);
	}
	
}
