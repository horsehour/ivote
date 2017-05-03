package com.horsehour.vote.rule.multiround;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

/**
 * General preference profile, including all preference rankings, and their
 * corresponding votes received from voters
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 10:05:21 PM, Dec 22, 2016
 **/
public class PrefProfile {
	public List<List<Position>> prefs;
	public List<Integer> votes;
	public List<Integer> items;
	public int numVoteTotal;

	public PrefProfile(List<List<Position>> prefs, List<Integer> votes, List<Integer> items) {
		this.prefs = new ArrayList<>(prefs);
		this.votes = new ArrayList<>(votes);
		if (items == null || items.size() == 0)
			this.items = getItems();
		else
			this.items = new ArrayList<>(items);
		this.numVoteTotal = 0;
		for (int i = 0; i < votes.size(); i++)
			this.numVoteTotal += votes.get(i);
	}

	/**
	 * @return all stated items
	 */
	public List<Integer> getItems() {
		List<Integer> items = new ArrayList<>();
		for (List<Position> pref : prefs)
			for (Position position : pref)
				items.addAll(position.items);
		items = items.stream().distinct().sorted().collect(Collectors.toList());
		return items;
	}

	/**
	 * @param item
	 * @return all positions of an item on all preference rankings, the position
	 *         will be -1 if it is not stated
	 */
	public List<Integer> getPositions(int item) {
		List<Integer> positions = new ArrayList<>();
		for (List<Position> pref : prefs) {
			for (Position position : pref)
				if (position.items.indexOf(item) == -1)
					positions.add(-1);
				else
					positions.add(position.val);
		}
		return positions;
	}

	/**
	 * Eliminate an item from given profile
	 * 
	 * @param item
	 * @return preference profile without given item
	 */
	public PrefProfile eliminate(int item) {
		List<List<Position>> preferences = new ArrayList<>(prefs.size());
		for (List<Position> pref : prefs) {
			List<Position> preference = new ArrayList<>(pref.size());
			for (Position pos : pref)
				preference.add(new Position(pos.val, pos.items));
			preferences.add(preference);
		}
		List<Integer> voteList = new ArrayList<>();

		List<Integer> rmPref = new ArrayList<>();
		for (int i = 0; i < prefs.size(); i++) {
			List<Position> pref = preferences.get(i);

			int d = 0, rmPosition = -1;
			for (int j = 0; j < pref.size(); j++) {
				Position position = pref.get(j);
				if (d == -1) {
					position.move(d);
					continue;
				}

				int idx = position.items.indexOf(item);
				if (idx == -1)
					continue;

				if (idx > -1) {
					// only one remaining item
					if (position.items.size() == 1) {
						rmPosition = j;
						d = -1;
						continue;
					} else
						position.items.remove(idx);
				}
			}

			// remove the position and the items
			if (rmPosition > -1)
				pref.remove(rmPosition);

			if (pref.size() > 0)
				voteList.add(votes.get(i));
			else
				rmPref.add(i);
		}

		Collections.reverse(rmPref);
		for (int i = 0; i < rmPref.size(); i++)
			preferences.remove((int) rmPref.get(i));

		List<Integer> itemList = new ArrayList<>(items);
		int index = itemList.indexOf(item);
		if (index > -1)
			itemList.remove(index);
		else
			System.err.println("ERROR: Failed to find item : " + item);
		return new PrefProfile(preferences, voteList, itemList);
	}

	/**
	 * Rebuild a preference profile with given items.
	 * 
	 * @param items
	 * @return rebuilt preference profile
	 */
	public PrefProfile rebuild(List<Integer> items) {
		List<List<Position>> preferences = new ArrayList<>();
		List<Integer> voteList = new ArrayList<>();

		Map<Integer, List<Integer>> map = null;

		List<Position> pref;
		for (int i = 0; i < prefs.size(); i++) {
			pref = prefs.get(i);

			List<Position> preference = new ArrayList<>();
			map = new TreeMap<>();
			for (int item : items) {
				for (Position pos : pref) {
					if (pos.items.indexOf(item) > -1) {
						List<Integer> list = null;
						if (map.containsKey(pos.val))
							list = map.get(pos.val);
						else
							list = new ArrayList<>();
						list.add(item);
						map.put(pos.val, list);
						break;
					}
				}
			}

			// corresponding preference ranking is totally eliminated
			if (map.isEmpty())
				continue;
			else
				voteList.add(votes.get(i));

			// possible tied items
			int c = 0;
			for (int key : map.keySet()) {
				preference.add(new Position(c, map.get(key)));
				c++;
			}
			preferences.add(preference);
		}
		return new PrefProfile(preferences, voteList, items);
	}

	/**
	 * Rebuild the profile with given items included (inclusive = true) or
	 * excluded (inclusive = false)
	 * 
	 * @param all
	 * @param items
	 * @param inclusive
	 * @return rebuilt preference profile
	 */
	public PrefProfile rebuild(List<Integer> all, List<Integer> items, boolean inclusive) {
		if (inclusive)
			return rebuild(items);

		List<Integer> remain = new ArrayList<>();
		for (int item : all)
			if (items.indexOf(item) == -1)
				remain.add(item);
		return rebuild(remain);
	}
}
