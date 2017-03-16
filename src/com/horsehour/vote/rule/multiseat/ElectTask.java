package com.horsehour.vote.rule.multiseat;

import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;

import com.horsehour.vote.DataEngine;
import com.horsehour.vote.Profile;

/**
 *
 * @author Chunheng Jiang
 * @version 1.0
 * @since 3:35:27 PM, Dec 30, 2016
 */

public class ElectTask implements Callable<Void> {
	public int id = -1;

	public List<Path> srcFiles = null;
	public Path outputFile = null;

	public ElectTask() {
		this.srcFiles = new ArrayList<>();
	}

	public ElectTask(Path destFile) {
		this.outputFile = destFile;
		this.srcFiles = new ArrayList<>();
	}

	public void addFile(Path srcFile) {
		this.srcFiles.add(srcFile);
	}

	public void setID(int id) {
		this.id = id;
	}

	@Override
	public Void call() throws Exception {
		Collections.shuffle(this.srcFiles);

		if (outputFile == null)
			outputFile = Paths.get("./output-" + id + ".txt");

		StringBuffer sb = new StringBuffer();
		String head = "Profile\tNumZeroL\tNumSingleL\tNumMultiL\tNumNodeWHLD\tNumNodeWOHLD\tNumNodeWESC\tNumNodeWOESC\tTime\tVictors\n";
		sb.append(head);

		OpenOption[] options = { StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE };

		STV rule = new STV();

		String name = "";
		for (Path socFile : srcFiles) {
			name = socFile.getFileName().toString();

			System.out.printf("%d - %s\n", id, name);
			sb.append(name);

			Profile<Integer> profile = DataEngine.loadProfile(socFile);
			List<Integer> winners = rule.getAllWinners(profile);

			String fmt = "\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.3f\t%s\n";
			sb.append(String.format(fmt, rule.numFailH, rule.numSingleL, rule.numMultiL, rule.numNodeWH,
					rule.numNodeWOH, rule.numNode, rule.numNodeFull, rule.time / 1000.0, winners));

			Files.write(outputFile, sb.toString().getBytes(), options);
			sb = new StringBuffer();
		}
		return null;
	}
}