package com.horsehour.util;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.io.FileUtils;

import com.google.common.io.Files;
import com.horsehour.vote.rule.Baldwin;
import com.horsehour.vote.rule.Black;
import com.horsehour.vote.rule.Borda;
import com.horsehour.vote.rule.Bucklin;
import com.horsehour.vote.rule.Condorcet;
import com.horsehour.vote.rule.Coombs;
import com.horsehour.vote.rule.Copeland;
import com.horsehour.vote.rule.InstantRunoff;
import com.horsehour.vote.rule.KemenyYoung;
import com.horsehour.vote.rule.Llull;
import com.horsehour.vote.rule.Maximin;
import com.horsehour.vote.rule.Nanson;
import com.horsehour.vote.rule.OklahomaVoting;
import com.horsehour.vote.rule.PairMargin;
import com.horsehour.vote.rule.Plurality;
import com.horsehour.vote.rule.RankedPairs;
import com.horsehour.vote.rule.Schulze;
import com.horsehour.vote.rule.Veto;
import com.horsehour.vote.rule.VotingRule;
import com.horsehour.vote.train.Eval1;

/**
 *
 * @author Chunheng Jiang
 * @version 1.0
 * @since 11:58:22 AM, Nov 26, 2016
 *
 */

public class TikzLog {
	public String tikzHead = "";
	public String tikzTail = "";
	public List<VotingRule> rules;
	public int numItem = 3;

	public TikzLog() {
		rules = new ArrayList<>();
		rules.add(new Baldwin());
		rules.add(new Black());
		rules.add(new Borda());
		rules.add(new Bucklin());
		rules.add(new Condorcet());
		rules.add(new Coombs());
		rules.add(new Copeland());
		rules.add(new InstantRunoff());
		rules.add(new KemenyYoung());
		rules.add(new Llull());
		rules.add(new Maximin());
		rules.add(new Nanson());
		rules.add(new OklahomaVoting());
		rules.add(new PairMargin());
		rules.add(new Plurality());
		rules.add(new RankedPairs());
		rules.add(new Schulze());
		rules.add(new Veto());

		getTikzHeadTail();
	}

	void getTikzHeadTail() {
		String headFile = "csc/ml/tikz.head";
		String tailFile = "csc/ml/tikz.tail";

		try {
			tikzHead = FileUtils.readFileToString(new File(headFile), "UTF8");
			tikzTail = FileUtils.readFileToString(new File(tailFile), "UTF8");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public double[][] getSimilarities(VotingRule oracle) {
		return getSimilarities(oracle, rules);
	}

	public double[][] getSimilarities(VotingRule oracle, List<VotingRule> rules){
		int[] numVotes = MathLib.Series.range(5, 2, 15);
		Eval1 eval = new Eval1();

		String name = oracle.toString();
		double[][] sim = new double[numVotes.length][rules.size()];

		VotingRule rule;
		for (int i = 0; i < numVotes.length; i++) {
			for (int j = 0; j < rules.size(); j++) {
				rule = rules.get(j);
				if (rule.toString().equals(name))
					sim[i][j] = 1;
				else
					sim[i][j] = eval.getSimilarity(numItem, numVotes[i], oracle, rule);
			}
		}
		return sim;
	}

	public void tikz(VotingRule oracle) throws IOException {
		String perfFile = "./csc/perf-mcp(r=" + oracle.toString() + ", m=" + numItem + ").txt";
		List<String> lines = Files.readLines(new File(perfFile), StandardCharsets.UTF_8);

		String name = oracle.toString();
		String groupplot = "\\nextgroupplot";
		String addplot = "\\addplot table [x = nv, y = %s,col sep=comma] {../data/%s.txt};";
		String addlegend = "\\addlegendentry{%s};";

		/**
		 * Prediction accuracies on n=[5,7,...,33], training accuracy, votes in
		 * sampling, number of samples, number of distinct samples, time
		 */
		int d = 20;
		List<double[]> data = new ArrayList<>();

		Pattern p = Pattern.compile("\\d+(\\.\\d+)?");

		StringBuffer tikz = new StringBuffer();
		tikz.append(tikzHead).append("\n");

		StringBuffer meta = new StringBuffer();
		int n = lines.size() / 3;
		double prev = 0;
		for (int i = 0; i < n; i++) {
			double[] entry = new double[d];
			Matcher m = p.matcher(lines.get(3 * i + 2));
			int c = 0;
			while (m.find()) {
				entry[c++] = Double.parseDouble(m.group());
			}

			m = p.matcher(lines.get(3 * i));
			while (m.find())
				entry[c++] = Double.parseDouble(m.group());
			data.add(entry);
			String label = "n" + entry[16] + "s" + entry[17];
			label = label.replaceAll("\\.0", "");
			meta.append("," + label);

			if (entry[16] != prev && entry[16] != 7) {
				tikz.append(groupplot).append("\n");
				prev = entry[16];
			}

			tikz.append(String.format(addplot, label, name)).append("\n");
			tikz.append(String.format(addlegend, label)).append("\n");
		}

		double[][] sim = getSimilarities(oracle);
		tikz.append(groupplot).append("\n");
		for (VotingRule rule : rules) {
			tikz.append(String.format(addplot, rule.toString(), name)).append("\n");
			tikz.append(String.format(addlegend, rule.toString())).append("\n");
		}
		/** we can compile all tex files at once: for i in *.tex; do pdflatex $i; done **/
		tikz.append(tikzTail).append("\n");
		FileUtils.writeStringToFile(new File("csc/ml/" + name + "-Train.tex"), tikz.toString(), "UTF8", false);

		StringBuffer sb = new StringBuffer();
		for (VotingRule rule : rules)
			meta.append(",").append(rule.toString());
		sb.append("nv").append(meta).append("\n");

		int[] numVotes = MathLib.Series.range(5, 2, 15);
		for (int i = 0; i < numVotes.length; i++) {
			int nv = numVotes[i];
			sb.append(nv);
			for (double[] entry : data)
				sb.append(",").append(entry[i]);
			for (double s : sim[i])
				sb.append(",").append(s);
			sb.append("\n");
		}
		FileUtils.writeStringToFile(new File("csc/ml/" + name + ".txt"), sb.toString(), "UTF8", false);
	}

	public void tikz() throws IOException {
		for (int i = 0; i < rules.size(); i++)
			tikz(rules.get(i));
	}

	public static void main(String[] args) throws Exception {
		TickClock.beginTick();

		// List<VotingRule> rules = new ArrayList<>();
		// rules.add(new Baldwin());
		// rules.add(new Black());
		// rules.add(new Borda());
		// rules.add(new Bucklin());
		// rules.add(new Condorcet());
		// rules.add(new Coombs());
		// rules.add(new Copeland());
		// rules.add(new InstantRunoff());
		// rules.add(new KemenyYoung());
		// rules.add(new Llull());
		// rules.add(new Maximin());
		// rules.add(new Nanson());
		// rules.add(new OklahomaVoting());
		// rules.add(new PairMargin());
		// rules.add(new Plurality());
		// rules.add(new RankedPairs());
		// rules.add(new Schulze());
		// rules.add(new Veto());

		TikzLog log = new TikzLog();
		log.tikz();
		
		TickClock.stopTick();
	}
}
