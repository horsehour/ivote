package com.horsehour.vote.train;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.IntSummaryStatistics;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiPredicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.Ace;
import com.horsehour.util.MulticoreExecutor;
import com.horsehour.util.TickClock;
import com.horsehour.vote.ChoiceTriple;
import com.horsehour.vote.DataEngine;
import com.horsehour.vote.Profile;
import com.horsehour.vote.axiom.CondorcetCriterion;
import com.horsehour.vote.axiom.ConsistencyCriterion;
import com.horsehour.vote.axiom.MonotonicityCriterion;
import com.horsehour.vote.axiom.NeutralityCriterion;
import com.horsehour.vote.axiom.VotingAxiom;
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

/**
 * Voting Rule Evaluation
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 2:49:12 PM, Jul 4, 2016
 */
public class Eval {
	/**
	 * @param numItem
	 * @param numVotes
	 * @param numSample
	 * @param oracle
	 * @return Random preference profiles
	 */
	List<List<ChoiceTriple<Integer>>> getSamples(int numItem, List<Integer> numVotes, int numSample,
			VotingRule oracle) {

		List<Pair<Integer, Integer>> aecs = new ArrayList<>();
		double sum = 0;
		for (int numVote : numVotes) {
			int numAEC = DataEngine.getAECStat(numItem, numVote).size();
			aecs.add(Pair.of(numVote, numAEC));
			sum += numAEC;
		}

		List<Callable<List<ChoiceTriple<Integer>>>> taskList = new ArrayList<>();
		final double ratio = numSample * 1.0 / sum;
		for (Pair<Integer, Integer> aec : aecs) {
			int numAEC = aec.getValue();
			int numVote = aec.getKey();
			int nSample = (int) (numAEC * ratio);
			if (nSample == 0) {
				continue;
			}
			taskList.add(() -> DataEngine.getRandomLabeledProfiles(numItem, numVote, nSample, oracle));
		}
		List<List<ChoiceTriple<Integer>>> samples = new ArrayList<>();
		try {
			samples = MulticoreExecutor.run(taskList);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return samples;
	}

	/**
	 * Generate Samples with Specific Parameters
	 * 
	 * @param numItem
	 * @param numVote
	 * @param numSampleList
	 * @param oracle
	 * @param axioms
	 * @return List of profiles and their choices
	 */
	List<List<ChoiceTriple<Integer>>> getSamples(int numItem, int numVote, List<Integer> numSampleList,
			VotingRule oracle, List<VotingAxiom> axioms) {
		int size = numSampleList.size();
		if (size <= 0) {
			System.err.println("ERROR: Please identify the accurate numbers of samples.");
			return null;
		}

		List<List<ChoiceTriple<Integer>>> samples = new ArrayList<>(size);
		samples.add(DataEngine.getRandomLabeledProfiles(numItem, numVote, numSampleList.get(0), oracle));

		// voting axioms are ignored
		if (size == 1)
			return samples;

		for (int i = 0; i < axioms.size(); i++) {
			VotingAxiom axiom = axioms.get(i);
			int numSample = numSampleList.get(i + 1);

			// Condorcet profiles
			if (axiom instanceof CondorcetCriterion) {
				samples.add(DataEngine.getRandomCondorcetProfiles(numItem, numVote, numSample));
				continue;
			}

			// consistent profiles
			if (axiom instanceof ConsistencyCriterion) {
				samples.add(DataEngine.getRandomConsistentProfiles(samples, numSample));
				continue;
			}

			// neutral profiles
			if (axiom instanceof NeutralityCriterion) {
				samples.add(DataEngine.getRandomNeutralProfiles(samples, numSample));
				continue;
			}

			// monotonic profiles
			if (axiom instanceof MonotonicityCriterion) {
				samples.add(DataEngine.getRandomMonotonicProfiles(samples, numSample));
				continue;
			}
		}
		return samples;
	}

	List<List<ChoiceTriple<Integer>>> getSamples(int numItem, int numVote, List<Integer> numSampleList,
			List<VotingRule> oracles, List<VotingAxiom> axioms) {
		int size = numSampleList.size();
		if (size <= 0) {
			System.err.println("ERROR: Please identify the accurate numbers of samples.");
			return null;
		}

		List<List<ChoiceTriple<Integer>>> samples = new ArrayList<>(size);
		// oracle voting rule labeled profiles
		int index = 0;
		for (VotingRule oracle : oracles)
			samples.add(DataEngine.getRandomLabeledProfiles(numItem, numVote, numSampleList.get(index++), oracle));

		// voting axioms are ignored
		if (size == oracles.size())
			return samples;

		for (int i = 0; i < axioms.size(); i++) {
			VotingAxiom axiom = axioms.get(i);
			int numSample = numSampleList.get(i + index);

			// Condorcet profiles
			if (axiom instanceof CondorcetCriterion) {
				samples.add(DataEngine.getRandomCondorcetProfiles(numItem, numVote, numSample));
				continue;
			}

			// consistent profiles
			if (axiom instanceof ConsistencyCriterion) {
				samples.add(DataEngine.getRandomConsistentProfiles(samples, numSample));
				continue;
			}

			// monotonic profiles
			if (axiom instanceof MonotonicityCriterion) {
				samples.add(DataEngine.getRandomMonotonicProfiles(samples, numSample));
				continue;
			}

			// neutral profiles
			if (axiom instanceof NeutralityCriterion) {
				samples.add(DataEngine.getRandomNeutralProfiles(samples, numSample));
				continue;
			}
		}
		return samples;
	}

	/**
	 * Single winner evaluation or comparison
	 */
	BiPredicate<List<Integer>, List<Integer>> match = (winners, predicted) -> {
		if (predicted == null)
			return false;
		return predicted.contains(winners.get(0));
	};

	/**
	 * @param labeledProfiles
	 * @param rule
	 * @return Prediction accuracy of the rule to the labeled profiles
	 */
	public double getPerformance(VotingRule rule, List<ChoiceTriple<Integer>> labeledProfiles) {
		AtomicLong numTotal = new AtomicLong(0), numMatch = new AtomicLong(0);
		labeledProfiles.stream().parallel().forEach(triple -> {
			List<Integer> winners = triple.getWinners();
			if (winners == null || winners.size() > 1)
				return;

			int num = triple.size();
			numTotal.addAndGet(num);

			Profile<Integer> profile = triple.getProfile();
			List<Integer> predicted = rule.getAllWinners(profile);
			// single winner
			if (match.test(winners, predicted))
				numMatch.addAndGet(num);
		});
		return numMatch.doubleValue() / numTotal.doubleValue();
	}

	/**
	 * Measure the goodness-of-fit of the learned rule to the expert rule
	 * 
	 * @param numItem
	 * @param numVote
	 * @param oracle
	 * @param learnedRule
	 * @return fitness
	 */
	public double getSimilarity(int numItem, int numVote, VotingRule oracle, VotingRule learnedRule) {
		Stream<Profile<Integer>> profiles = DataEngine.getAECPreferenceProfiles(numItem, numVote);

		AtomicLong numTotal = new AtomicLong(0), numMatch = new AtomicLong(0);
		profiles.forEach(profile -> {
			List<Integer> winners = oracle.getAllWinners(profile);
			// single winner
			if (winners == null || winners.size() > 1)
				return;

			// Long stat = profile.getStat();
			numTotal.addAndGet(1);

			List<Integer> predicted = learnedRule.getAllWinners(profile);
			if (match.test(winners, predicted))
				numMatch.addAndGet(1);
		});
		return numMatch.doubleValue() / numTotal.doubleValue();
	}

	public double getSimilarity(VotingRule oracle, VotingRule learnedRule, List<Profile<Integer>> profiles) {

		AtomicLong numTotal = new AtomicLong(0), numMatch = new AtomicLong(0);
		profiles.forEach(profile -> {
			List<Integer> winners = oracle.getAllWinners(profile);
			if (winners == null || winners.size() > 1)
				return;

			// Long stat = profile.getStat();
			numTotal.addAndGet(1);

			List<Integer> predicted = learnedRule.getAllWinners(profile);
			if (match.test(winners, predicted))
				numMatch.addAndGet(1);
		});
		return numMatch.doubleValue() / numTotal.doubleValue();
	}

	public int countSamples(List<ChoiceTriple<Integer>> profiles) {
		return profiles.stream().parallel().map(triple -> triple.size()).reduce(Integer::sum).get();
	}

	/**
	 * Bridge between data in different format
	 */
	public static class DataBridge {
		static SampleSet getSampleSet(double[][] features, int[] labels) {
			SampleSet sampleset = new SampleSet();
			for (int i = 0; i < labels.length; i++)
				sampleset.addSample(new Sample(features[i], labels[i]));
			return sampleset;
		}

		static SampleSet getSampleSet(Pair<double[][], int[]> pair) {
			return getSampleSet(pair.getKey(), pair.getValue());
		}

		static SampleSet getSampleSet(List<ChoiceTriple<Integer>> profiles) {
			return getSampleSet(DataEngine.getFlatDataSet(profiles));
		}

		static Pair<double[][], int[]> mergePairList(List<Pair<double[][], int[]>> pairList) {
			int nPair = pairList.size();
			int numTrain = 0;
			for (int i = 0; i < nPair; i++)
				numTrain += pairList.get(i).getLeft().length;
			int dim = pairList.get(0).getLeft()[0].length;

			double[][] trainset = new double[numTrain][dim];
			int[] labels = new int[numTrain];

			int count = 0;
			for (Pair<double[][], int[]> pair : pairList) {
				double[][] x = pair.getLeft();
				int[] y = pair.getRight();
				for (int i = 0; i < x.length; i++) {
					trainset[count] = x[i];
					labels[count] = y[i];
					count++;
				}
			}
			return Pair.of(trainset, labels);
		}

		static JavaRDD<LabeledPoint> getLabeledPoints(JavaSparkContext jsc, List<ChoiceTriple<Integer>> profiles) {
			Pair<double[][], int[]> dataset = DataEngine.getFlatDataSet(profiles);
			double[][] inputs = dataset.getKey();
			int[] labels = dataset.getValue();

			int nSample = inputs.length;
			List<LabeledPoint> list = IntStream.range(0, nSample).boxed()
					.map(i -> new LabeledPoint(labels[i], Vectors.dense(inputs[i]))).collect(Collectors.toList());
			return jsc.parallelize(list);
		}
	}

	/**
	 * @param numItem
	 * @param numVote
	 * @param numSample
	 * @param oracle
	 * @return learned voting rule
	 * @throws IOException
	 */
	public VotingRule learnRule(int numItem, int numVote, int numSample, VotingRule oracle) throws IOException {
		List<List<ChoiceTriple<Integer>>> samples = null;
		samples = getSamples(numItem, numVote, Arrays.asList(numSample), oracle, null);

		List<ChoiceTriple<Integer>> trainset = null;
		trainset = samples.stream().flatMap(list -> list.stream()).collect(Collectors.toList());

		Learner vrl = new Learner();
		VotingRule learnedRule = null;
		// learnedRule = vrl.getDecisionTreeRule(trainset);
		// learnedRule = vrl.getNeuralDecisionTreeRule(trainset, 3);
		learnedRule = vrl.getMaximumCutPlaneRule(trainset, 6);

		// learnedRule = vrl.getAdaBoostRule(trainset);
		// learnedRule = vrl.getLogisticRegressionRule(trainset);
		// learnedRule = vrl.getNeuralNetworkRule(trainset);
		// learnedRule = vrl.getRandomForestRule(trainset);

		return learnedRule;
	}

	public VotingRule learnRule(int numItem, List<Integer> numVotes, int numSample, VotingRule oracle)
			throws IOException {
		List<List<ChoiceTriple<Integer>>> samples = null;
		samples = getSamples(numItem, numVotes, numSample, oracle);

		List<ChoiceTriple<Integer>> trainset = null;
		trainset = samples.stream().flatMap(list -> list.stream()).collect(Collectors.toList());

		Learner vrl = new Learner();
		VotingRule learnedRule = null;
		// learnedRule = vrl.getDecisionTreeRule(trainset);
		// learnedRule = vrl.getNeuralDecisionTreeRule(trainset, 3);
		learnedRule = vrl.getMaximumCutPlaneRule(trainset, 6);

		// learnedRule = vrl.getAdaBoostRule(trainset);
		// learnedRule = vrl.getLogisticRegressionRule(trainset);
		// learnedRule = vrl.getNeuralNetworkRule(trainset);
		// learnedRule = vrl.getRandomForestRule(trainset);

		return learnedRule;
	}

	/**
	 * Learn rule and some axioms
	 * 
	 * @param numItem
	 * @param numVote
	 * @param maxNumVote
	 *            maximum number of voters for evaluation of generalization
	 * @param numSample
	 * @param oracle
	 * @param axiomIndicesTrain
	 *            axioms used in training, empty indicates that no axiom will be
	 *            used during training
	 * @param axioms
	 *            axioms used for evaluation
	 * 
	 * @return learned voting rule satisifying specific axioms
	 */
	public VotingRule learnRuleAndAxioms(int numItem, int numVote, int maxNumVote, int numSample,
			int[] axiomIndicesTrain, VotingRule oracle, List<VotingAxiom> axioms) {
		List<Integer> numSamples = new ArrayList<>(axioms.size() + 1);
		numSamples.add(numSample);
		List<VotingAxiom> axiomTrain = new ArrayList<>(axiomIndicesTrain.length);
		for (int i = 0; i < axiomIndicesTrain.length; i++) {
			axiomTrain.add(axioms.get(i));
			numSamples.add(numSample);
		}

		List<List<ChoiceTriple<Integer>>> samples = getSamples(numItem, numVote, numSamples, oracle, axiomTrain);

		List<ChoiceTriple<Integer>> trainset = samples.stream().flatMap(list -> list.stream())
				.collect(Collectors.toList());

		Learner vrl = new Learner();
		VotingRule learnedRule = null;

		// learnedRule = vrl.getKNNRule(trainset);
		// learnedRule = vrl.getNaiveBayesRule(trainset);
		//
		// learnedRule = vrl.getLogisticRegressionRule(trainset);
		//
		// learnedRule = vrl.getFLDRule(trainset);
		// learnedRule = vrl.getLDARule(trainset);
		// learnedRule = vrl.getQDARule(trainset);
		// learnedRule = vrl.getRDARule(trainset);
		//
		// learnedRule = vrl.getAdaBoostRule(trainset);
		//
		// learnedRule = vrl.getRBFNetworkRule(trainset);
		// learnedRule = vrl.getSVMRule(trainset);
		// learnedRule = vrl.getSVMEnsembleRule(trainset);

		// learnedRule = vrl.getDecisionTreeRule(trainset);
		// learnedRule = vrl.getRandomForestRule(trainset);
		// learnedRule = vrl.getScalaRandomForestRule(trainset);
		// learnedRule = vrl.getGradientTreeBoostRule(trainset, 3, 5);

		// learnedRule = vrl.getNeuralNetworkRule(trainset);
		// learnedRule = vrl.getDeepLearningRule(trainset);
		// learnedRule = vrl.getCNNRule(trainset);
		// learnedRule = vrl.getDLRFEnsembleRule(trainset);

		// List<LearnedRule> learnedRules =
		// vrl.getDecomposedRandomForestRule(trainset);

		// learnedRule = vrl.getMultiDecisionTreeRule(trainset, 3);
		// learnedRule = vrl.getMDTRule(trainset);
		// learnedRule = vrl.getRankNetRule(trainset);

		// learnedRule = vrl.getPermutedDTRule(trainset, 3);
		// learnedRule = vrl.getPermutedMDTRule(trainset, 3);
		// learnedRule = vrl.getPermutedLRRule(trainset);
		// learnedRule = vrl.getPermutedDLRule(trainset);
		// learnedRule = vrl.getPermutedNNRule(trainset);
		// learnedRule = vrl.getPermutedMDTNNRule(trainset, 3);
		// learnedRule = vrl.getPermutedMDTPermutedNNRule(trainset, 3);
		// learnedRule = vrl.getPermutedDTPermutedNNRule(trainset, 3);
		learnedRule = vrl.getPermutedDTPermutedLRRule(trainset, 3);

		getBriefShape(samples);

		try {
			evalTrainPerfAndGeneralization(numItem, numVote, maxNumVote, axiomIndicesTrain, samples, learnedRule,
					oracle, axioms);
		} catch (IOException e) {
			e.printStackTrace();
		}

		return learnedRule;
	}

	/**
	 * Learn rules and some axioms, we expect that the learned rule has high
	 * satisfiabilities of those axioms met by the ground truth voting rules
	 * 
	 * @param numItem
	 * @param numVote
	 * @param maxNumVote
	 *            maximum number of voters for evaluation of generalization
	 * @param numSample
	 * @param oracles
	 *            oracles used in training
	 * @param trainAxiomIndices
	 *            axioms used in training, empty indicates that no axiom will be
	 *            used during training
	 * @param axioms
	 *            axioms used for evaluation
	 * 
	 * @return learned voting rules which satisfying specific axioms
	 */
	public VotingRule learnRulesAndAxioms(int numItem, int numVote, int maxNumVote, int numSample,
			int[] trainAxiomIndices, List<VotingRule> oracles, List<VotingAxiom> axioms) {

		StringBuffer sb = new StringBuffer();

		List<Integer> numSamples = new ArrayList<>(oracles.size() + axioms.size());
		for (int i = 0; i < oracles.size(); i++) {
			numSamples.add(numSample);
			sb.append(oracles.get(i).toString() + "-");
		}

		List<VotingAxiom> axiomTrain = new ArrayList<>(trainAxiomIndices.length);
		for (int index : trainAxiomIndices) {
			axiomTrain.add(axioms.get(index));
			numSamples.add(numSample);
			sb.append(axioms.get(index).toString() + "-");
		}

		List<List<ChoiceTriple<Integer>>> samples = getSamples(numItem, numVote, numSamples, oracles, axiomTrain);

		List<ChoiceTriple<Integer>> trainset = samples.stream().flatMap(list -> list.stream())
				.collect(Collectors.toList());

		Learner vrl = new Learner(sb.toString());

		VotingRule learnedRule = vrl.getPermutedDTPermutedLRRule(trainset, 3);
		// VotingRule learnedRule = vrl.getNeuralDecisionTreeRule(trainset, 3);
		// getSummaryOfTrainSet(samples);

		try {
			evalTrainPerfAndGeneralization(numItem, numVote, maxNumVote, trainAxiomIndices, samples, learnedRule,
					oracles, axioms);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return learnedRule;
	}

	/**
	 * summary of the samples information
	 * 
	 * @param samples
	 */
	public void getBriefShape(List<List<ChoiceTriple<Integer>>> samples) {
		int nItem = samples.get(0).get(0).getProfile().getNumItem();
		int nKindSample = samples.size();
		// distribution of profiles with different number of winners
		double[][] szTiedVictories = new double[nKindSample][nItem];
		for (int i = 0; i < nKindSample; i++) {
			Map<Integer, Integer> cluster = samples.get(i).stream().map(t -> Pair.of(t.getWinners().size(), t.size()))
					.collect(Collectors.groupingBy(p -> p.getLeft(), Collectors.summingInt(p -> p.getRight())));
			int total = samples.get(i).stream().map(t -> t.size()).reduce(Integer::sum).get();
			for (int k : cluster.keySet())
				szTiedVictories[i][k - 1] = cluster.get(k).doubleValue() / total;
		}

		List<String> rowLabels = new ArrayList<>();
		for (int i = 0; i < nKindSample; i++)
			rowLabels.add("kind-" + (i + 1));
		List<String> colLabels = new ArrayList<>();
		for (int i = 0; i < nItem; i++)
			colLabels.add("size = " + (i + 1));
		Ace ace = new Ace("Distribution of Profiles with Different Num of Winners");
		ace.bar(rowLabels, colLabels, szTiedVictories);

		/**
		 * compute the lower bound of training error (sample profile, different
		 * labels) or the upper bound of prediction accuracy for one single
		 * winner rule
		 */
		List<ChoiceTriple<Integer>> trainset = samples.stream().flatMap(list -> list.stream())
				.collect(Collectors.toList());

		List<Integer> hashCodeList = new ArrayList<>();
		List<int[]> labelCountTable = new ArrayList<>();
		for (ChoiceTriple<Integer> triple : trainset) {
			int hashCode = triple.getProfile().hashCode();
			int index = hashCodeList.indexOf(hashCode);
			int[] labelCount = null;
			if (index == -1) {
				hashCodeList.add(hashCode);
				labelCount = new int[nItem];
				labelCountTable.add(labelCount);
			} else
				labelCount = labelCountTable.get(index);

			for (int winner : triple.getWinners())
				labelCount[winner] += triple.size();
		}

		double[] labelDistribution = new double[nItem];
		long bound = 0, total = 0;
		for (int[] labelCount : labelCountTable) {
			IntSummaryStatistics summary = Arrays.stream(labelCount).summaryStatistics();
			bound += summary.getMax();
			total += summary.getSum();
			for (int i = 0; i < nItem; i++)
				labelDistribution[i] += labelCount[i];
		}

		List<String> columnLabels = new ArrayList<>();
		for (int i = 0; i < nItem; i++) {
			columnLabels.add("Cand. " + (i + 1));
		}
		StringBuffer meta = new StringBuffer();
		meta.append("Training Set Label Distribution (Upper Bound. Pred Accuracy = ");
		meta.append(String.format(".2f%, Number of Kinds of Profiles = %d)", bound * 100.0 / total,
				labelCountTable.size()));

		ace = new Ace(meta.toString());
		ace.pie(columnLabels, labelDistribution);
	}

	/**
	 * Evaluate training performance and generalization ability about the
	 * learned rule
	 * 
	 * @param numItem
	 * @param numVote
	 * @param maxNumVote
	 * @param axiomIndicesTrain
	 * @param samples
	 * @param learnedRule
	 * @param oracle
	 * @param axioms
	 * @throws IOException
	 */
	public void evalTrainPerfAndGeneralization(int numItem, int numVote, int maxNumVote, int[] axiomIndicesTrain,
			List<List<ChoiceTriple<Integer>>> samples, VotingRule learnedRule, VotingRule oracle,
			List<VotingAxiom> axioms) throws IOException {

		StringBuffer sb = new StringBuffer();
		sb.append(" ===== " + learnedRule.toString() + " =====\n");

		String format = "%-10s\t%-10sDetails\n";
		sb.append(String.format(format, "LearnedRule", oracle.toString()));

		// 1 (training performance) + 1 (goodness of fit) + axioms
		int numGroup = 2 + axioms.size();

		List<double[][]> perfList = new ArrayList<>(numGroup);
		List<List<String>> rowLabelList = new ArrayList<>(numGroup);
		List<List<String>> columnLabelList = new ArrayList<>(numGroup);

		int nRow = 2, nCol = 2 + axiomIndicesTrain.length;
		double[][] data = new double[nRow][nCol];

		// 1 (labeled profiles) + 1 (overall training perf) + axioms
		List<String> metaList = new ArrayList<>(numGroup);
		List<String> columnLabel = new ArrayList<>(numGroup);

		metaList.add("-- Training Accuracy (" + oracle.toString() + " Data)");
		columnLabel.add(oracle.toString());

		for (int i : axiomIndicesTrain) {
			String shortName = axioms.get(i).toString().replace("Criterion", "");
			metaList.add("-- Training Accuracy (" + shortName + " Data)");
			columnLabel.add(shortName);
		}
		metaList.add("-- Overall Training Accuracy");
		columnLabel.add("Overall");

		format = "%1.3f\t\t%1.3f\t%15s\t%21s";

		List<ChoiceTriple<Integer>> sample = null;
		for (int i = 0; i <= samples.size(); i++) {
			if (i == samples.size())// overall
				sample = samples.stream().flatMap(list -> list.stream()).collect(Collectors.toList());
			else
				sample = samples.get(i);
			data[0][i] = getPerformance(learnedRule, sample);
			data[1][i] = getPerformance(oracle, sample);

			int count = countSamples(sample);
			sb.append(String.format(format, data[0][i], data[1][i], metaList.get(i), count + "\n"));
			columnLabel.set(i, columnLabel.get(i) + " : " + count);
		}
		perfList.add(data);
		rowLabelList.add(Arrays.asList("Learned Rule", oracle.toString()));
		columnLabelList.add(columnLabel);

		int index = 0;
		int len = 1 + (maxNumVote - numVote) / 2;

		nRow = 1;
		nCol = len;
		data = new double[nRow][nCol];
		columnLabel = new ArrayList<>(len);

		format = "%1$1.3f\t\t%2$1.3f\t%3$-45s\t%4$-10s";
		for (int i = numVote; i <= maxNumVote; i += 2, index++) {
			data[0][index] = getSimilarity(numItem, i, oracle, learnedRule);
			sb.append(String.format(format, data[0][index], 1.0, "-- Fitness to " + oracle.toString(),
					"m = " + numItem + ", n = " + i + "\n"));
			columnLabel.add("n = " + i);
		}
		perfList.add(data);
		rowLabelList.add(Arrays.asList(""));
		columnLabelList.add(columnLabel);

		nRow = 2;
		nCol = len;

		for (VotingAxiom axiom : axioms) {
			data = new double[nRow][nCol];
			index = 0;
			for (int i = numVote; i <= maxNumVote; i += 2, index++) {
				data[0][index] = axiom.getSatisfiability(numItem, i, learnedRule);
				data[1][index] = axiom.getSatisfiability(numItem, i, oracle);
				sb.append(String.format(format, data[0][index], data[1][index], "-- Satness to " + axiom.toString(),
						"m = " + numItem + ", n = " + i + "\n"));
			}
			perfList.add(data);
			rowLabelList.add(Arrays.asList("LearnedRule", oracle.toString()));
			columnLabelList.add(columnLabelList.get(columnLabelList.size() - 1));
		}

		FileUtils.write(new File("csc/model.txt"), sb.toString(), "UTF8", true);

		String outputFile = "./csc/report/" + learnedRule.toString() + "-m" + numItem + ".png";
		Ace ace = new Ace("Learning Voting Rule (m = " + numItem + ")", outputFile);
		ace.combinedRangeBars(rowLabelList, columnLabelList, "", "SAT", perfList);
	}

	/**
	 * Evaluate training performance and generalization ability about the
	 * learned rule
	 * 
	 * @param numItem
	 * @param numVote
	 * @param maxNumVote
	 * @param trainAxiomIndices
	 * @param samples
	 * @param learnedRule
	 * @param oracles
	 * @param axioms
	 * @throws IOException
	 */
	public void evalTrainPerfAndGeneralization(int numItem, int numVote, int maxNumVote, int[] trainAxiomIndices,
			List<List<ChoiceTriple<Integer>>> samples, VotingRule learnedRule, List<VotingRule> oracles,
			List<VotingAxiom> axioms) throws IOException {

		// 1 (training performance) + oracles (goodness of fit) + axioms
		int numGroup = 1 + oracles.size() + axioms.size();

		List<double[][]> perfList = new ArrayList<>(numGroup);
		List<List<String>> rowLabelList = new ArrayList<>(numGroup);
		List<List<String>> columnLabelList = new ArrayList<>(numGroup);

		// learned rule and oracles
		int nRow = 1 + oracles.size();
		// oracles-labeled, training axioms-related and overall profiles
		int nCol = oracles.size() + trainAxiomIndices.length + 1;
		double[][] data = new double[nRow][nCol];

		List<String> columnLabel = new ArrayList<>(numGroup);
		for (VotingRule oracle : oracles)
			columnLabel.add(oracle.toString());

		for (int index : trainAxiomIndices) {
			String shortName = axioms.get(index).toString().replace("Criterion", "");
			columnLabel.add(shortName);
		}
		columnLabel.add("Overall");

		List<ChoiceTriple<Integer>> sample = null;
		for (int i = 0; i <= samples.size(); i++) {
			if (i == samples.size())// overall
				sample = samples.stream().flatMap(list -> list.stream()).collect(Collectors.toList());
			else
				sample = samples.get(i);

			data[0][i] = getPerformance(learnedRule, sample);
			for (int k = 0; k < oracles.size(); k++) {
				data[k + 1][i] = getPerformance(oracles.get(k), sample);
			}

			int count = countSamples(sample);
			columnLabel.set(i, columnLabel.get(i) + " : " + count);
		}
		perfList.add(data);
		List<String> rowLabels = new ArrayList<>();
		rowLabels.add("Learned Rule");

		List<String> oracleNameList = new ArrayList<>(oracles.size());
		for (VotingRule oracle : oracles)
			oracleNameList.add(oracle.toString());

		rowLabels.addAll(oracleNameList);
		rowLabelList.add(rowLabels);
		columnLabelList.add(columnLabel);

		int len = 1 + (maxNumVote - numVote) / 2;

		nRow = oracles.size();
		nCol = len;
		data = new double[nRow][nCol];
		columnLabel = new ArrayList<>(len);

		int index = 0;
		for (int i = numVote; i <= maxNumVote; i += 2, index++) {
			columnLabel.add("n = " + i);
			for (int k = 0; k < oracles.size(); k++)
				data[k][index] = getSimilarity(numItem, i, oracles.get(k), learnedRule);
		}
		perfList.add(data);
		rowLabelList.add(oracleNameList);
		columnLabelList.add(columnLabel);

		nRow = oracles.size() + 1;
		nCol = len;

		for (VotingAxiom axiom : axioms) {
			data = new double[nRow][nCol];
			index = 0;
			for (int i = numVote; i <= maxNumVote; i += 2, index++) {
				data[0][index] = axiom.getSatisfiability(numItem, i, learnedRule);
				for (int k = 0; k < oracles.size(); k++)
					data[k + 1][index] = axiom.getSatisfiability(numItem, i, oracles.get(k));
			}
			perfList.add(data);
			rowLabelList.add(rowLabelList.get(0));
			columnLabelList.add(columnLabelList.get(columnLabelList.size() - 1));
		}

		String outputFile = "./csc/report/" + learnedRule.toString() + "-m" + numItem + ".png";
		Ace ace = new Ace("Learning Voting Rule (m = " + numItem + ")", outputFile);
		ace.combinedRangeBars(rowLabelList, columnLabelList, "", "SAT", perfList);
	}

	public void getSimilarity(int numItem, int[] numVotes, List<VotingRule> rules) {
		int n = rules.size();

		List<String> seriesLabel = new ArrayList<>();
		for (int i = 0; i < n; i++)
			seriesLabel.add(rules.get(i).toString());

		double[] xData = new double[numVotes.length];
		for (int i = 0; i < numVotes.length; i++)
			xData[i] = numVotes[i];

		for (int i = 0; i < n; i++) {
			VotingRule rule = rules.get(i);
			Ace ace = new Ace(rule.toString());
			double[][] sim = new double[numVotes.length][n];
			for (int k = 0; k < numVotes.length; k++) {
				int numVote = numVotes[k];
				for (int j = 0; j < n; j++) {
					if (i == j)
						sim[k][j] = 1.0;
					else
						sim[k][j] = getSimilarity(numItem, numVote, rule, rules.get(j));
				}
			}
			ace.combinedLines("n", seriesLabel, xData, sim);
		}
	}

	/**
	 * Evaluate the learned rule on random samples
	 * 
	 * @param numItem
	 * @param numVote
	 * @param maxNumVote
	 * @param numSample
	 * @param learnedRule
	 * @param oracle
	 * @param axioms
	 */
	public void evalLearnedRule(int numItem, int numVote, int maxNumVote, int numSample, VotingRule learnedRule,
			VotingRule oracle, List<VotingAxiom> axioms) {

		List<String> nameList = new ArrayList<>(axioms.size());
		for (VotingAxiom axiom : axioms)
			nameList.add(axiom.toString().replace("Criterion", ""));

		int numRecords = 1 + (maxNumVote - numVote) / 2;

		double[] xData = new double[numRecords];
		double[][] yData = new double[numRecords][2 * axioms.size() + 1];

		List<Profile<Integer>> profiles = null;
		List<String> seriesLabel = new ArrayList<>();
		int index = 0;
		for (int v = numVote; v <= maxNumVote; v += 2, index++) {
			profiles = DataEngine.getRandomProfiles(numItem, v, numSample);
			xData[index] = v;

			yData[index][0] = getSimilarity(oracle, learnedRule, profiles);
			seriesLabel.add("Goodness to Fit " + oracle.toString());

			for (int i = 0; i < axioms.size(); i++) {
				yData[index][2 * i + 1] = axioms.get(i).getSatisfiability(profiles, oracle);
				yData[index][2 * i + 2] = axioms.get(i).getSatisfiability(profiles, learnedRule);

				seriesLabel.add(oracle.toString() + "_" + nameList.get(i));
				seriesLabel.add("Learned_" + nameList.get(i));
			}
		}
		Ace ace = new Ace(learnedRule.toString() + " (nItem = " + numItem + ", nSample = " + numSample + ")");
		ace.combinedLines("Number of Voters", seriesLabel, xData, yData);
	}

	/**
	 * Evaluate the satisfiabilities of rules to some axioms
	 * 
	 * @param numItem
	 * @param numVote
	 * @param maxNumVote
	 * @param rule
	 * @param axioms
	 */
	public void evalSatToAxiom(int numItem, int numVote, int maxNumVote, VotingRule rule, List<VotingAxiom> axioms) {
		int numRecords = 1 + maxNumVote - numVote;

		List<String> seriesLabel = new ArrayList<>();
		for (VotingAxiom axiom : axioms)
			seriesLabel.add(axiom.toString().replace("Criterion", ""));

		List<Profile<Integer>> profiles = null;

		double[] xData = new double[numRecords];
		double[][] yData = new double[numRecords][axioms.size()];

		int index = 0;
		for (int v = numVote; v <= maxNumVote; v++, index++) {
			xData[index] = v;
			profiles = DataEngine.getAECPreferenceProfiles(numItem, v).collect(Collectors.toList());
			for (int k = 0; k < axioms.size(); k++) {
				VotingAxiom axiom = axioms.get(k);
				yData[index][k] = axiom.getSatisfiability(profiles, rule);
			}
		}

		Ace ace = new Ace(rule.toString() + " SAT to Some Axioms (m = " + numItem + ")");
		ace.lines("Number of Voters", "Satisfiability", seriesLabel, xData, yData);
	}

	/**
	 * Evaluate the satisfiabilities of some voting rules to some voting axioms
	 * 
	 * @param numItem
	 * @param numVote
	 * @param rules
	 * @param axioms
	 */
	public void evalSatToAxioms(int numItem, int numVote, Collection<VotingRule> rules, List<VotingAxiom> axioms) {
		List<String> xTickLabelList = new ArrayList<>(), yTickLabelList = new ArrayList<>();
		for (VotingRule rule : rules)
			yTickLabelList.add(rule.toString());

		List<Profile<Integer>> profiles = DataEngine.getAECPreferenceProfiles(numItem, numVote)
				.collect(Collectors.toList());

		double[][] satMatrix = new double[axioms.size()][rules.size()];
		for (int i = 0; i < axioms.size(); i++) {
			VotingAxiom axiom = axioms.get(i);
			xTickLabelList.add(axiom.toString().replace("Criterion", ""));
			int j = 0;
			for (VotingRule rule : rules)
				satMatrix[i][j++] = axiom.getSatisfiability(profiles, rule);
		}
		String outputFile = "/Users/chjiang/Documents/workspace/java/ivote/csc/report/SATTable" + "-m" + numItem
				+ ".png";
		Ace ace = new Ace("SAT to Some Axioms (m = " + numItem + ", n = " + numVote + ")", outputFile);
		ace.heatmap("", "", xTickLabelList, yTickLabelList, satMatrix);
	}

	/**
	 * Evaluate the satisfiabilities of voting rules to some axioms
	 * 
	 * @param numItem
	 * @param numVote
	 * @param maxNumVote
	 * @param rules
	 * @param axioms
	 */
	public void evalSatToAxioms(int numItem, int numVote, int maxNumVote, Collection<VotingRule> rules,
			List<VotingAxiom> axioms) {
		int numRecords = 1 + maxNumVote - numVote;

		List<String> seriesLabel = new ArrayList<>();
		for (VotingAxiom axiom : axioms)
			seriesLabel.add(axiom.toString().replace("Criterion", ""));

		double[] xData = new double[numRecords];
		List<double[][]> yDataList = new ArrayList<>();
		for (int r = 0; r < rules.size(); r++)
			yDataList.add(new double[numRecords][axioms.size()]);

		List<Profile<Integer>> profiles = null;
		int index = 0;
		for (int v = numVote; v <= maxNumVote; v++, index++) {
			xData[index] = v;
			profiles = DataEngine.getAECPreferenceProfiles(numItem, v).collect(Collectors.toList());
			int r = -1;
			for (VotingRule rule : rules) {
				r++;
				for (int k = 0; k < axioms.size(); k++) {
					VotingAxiom axiom = axioms.get(k);
					yDataList.get(r)[index][k] = axiom.getSatisfiability(profiles, rule);
				}
			}
		}
		Ace ace = new Ace("SAT to Some Axioms (m = " + numItem + ")");
		ace.grid("Number of Voters", seriesLabel, xData, yDataList, rules.size(), 1);
	}

	/**
	 * Evaluate the satisfiabilities of rules to some axioms with random
	 * sampling
	 * 
	 * @param numItem
	 * @param numVote
	 * @param maxNumVote
	 * @param numSample
	 * @param rule
	 * @param axioms
	 */
	public void evalSatToAxioms(int numItem, int numVote, int maxNumVote, int numSample, VotingRule rule,
			List<VotingAxiom> axioms) {
		int numRecords = 1 + (maxNumVote - numVote);

		int index = 0;
		double[] xData = new double[numRecords];
		for (int v = numVote; v <= maxNumVote; v++, index++)
			xData[index] = v;

		List<String> seriesLabel = new ArrayList<>();
		double[][] yData = new double[numRecords][axioms.size()];
		for (int k = 0; k < axioms.size(); k++) {
			VotingAxiom axiom = axioms.get(k);
			List<Profile<Integer>> profiles = null;
			index = 0;
			for (int v = numVote; v <= maxNumVote; v++, index++) {
				profiles = DataEngine.getRandomProfiles(numItem, v, numSample);
				yData[index][k] = axiom.getSatisfiability(profiles, rule);
			}
			seriesLabel.add(axiom.toString().replace("Criterion", ""));
		}

		Ace ace = new Ace(rule.toString() + " SAT to Some Axioms (m = " + numItem + ")");
		ace.lines("Number of Voters", "Satisfiability", seriesLabel, xData, yData);
	}

	public static void main(String[] args) throws Exception {
		TickClock.beginTick();

		// List<VotingAxiom> axioms = new ArrayList<>();
		// axioms.add(new CondorcetCriterion());
		// axioms.add(new CondorcetLoserCriterion());
		// axioms.add(new ConsistencyCriterion());
		// axioms.add(new HomogeneityCriterion());
		// axioms.add(new MajorityCriterion());
		// axioms.add(new MajorityLoserCriterion());
		// axioms.add(new MonotonicityCriterion());
		// axioms.add(new NeutralityCriterion());
		// axioms.add(new ParticipationCriterion());
		// axioms.add(new PluralityCriterion());
		// axioms.add(new ReversalSymmetryCriterion());
		// axioms.add(new SmithCriterion());
		// axioms.add(new SchwarzCriterion());

		Eval eval = new Eval();

		List<VotingRule> rules = new ArrayList<>();
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

		int numItem = 3, maxNumVote = 21, numSample = 500;

		// eval.getSimilarity(numItem, MathLib.Series.range(5, 2, 15), rules);

		List<Integer> numVotes = new ArrayList<>();
		for (int numVote = 5; numVote <= maxNumVote; numVote += 2)
			numVotes.add(numVote);

		List<Callable<VotingRule>> learnTaskList = new ArrayList<>();
		for (int i = 0; i < rules.size(); i++) {
			VotingRule oracle = rules.get(i);
			learnTaskList.add(() -> eval.learnRule(numItem, numVotes, numSample, oracle));
		}

		List<VotingRule> learnedRules = MulticoreExecutor.run(learnTaskList);
		VotingRule oracle;
		StringBuffer sb = new StringBuffer();
		sb.append("nv");
		for (int numVote : numVotes)
			sb.append(String.format("\t%d", numVote));
		sb.append("\n");

		for (int i = 0; i < rules.size(); i++) {
			oracle = rules.get(i);
			sb.append(oracle.toString());
			for (int numVote : numVotes) {
				double prec = eval.getSimilarity(numItem, numVote, oracle, learnedRules.get(i));
				sb.append(String.format("\t%.2f", prec));
			}
			sb.append("\n");
		}
		FileUtils.writeStringToFile(new File("csc/MaxCutPlane.txt"), sb.toString(), "UTF8", true);

		TickClock.stopTick();
	}
}