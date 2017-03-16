package com.horsehour.vote;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

import com.horsehour.util.TickClock;

/**
 * Selection with replacement (SWR) to generate the index form for the named
 * preference profiles (NPP) or all the anonymous equivalent class (AEC). If the
 * items used to produce random sample is equal to the times to sample, then
 * it's the well-known bootstrapping method.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 1:43:26 PM, Jun 22, 2016
 * @see EĞECIOĞLU Ö, Giritligil A E. The Impartial, Anonymous, and Neutral
 *      Culture Model: A Probability Model for Sampling Public Preference
 *      Structures[J]. The Journal of Mathematical Sociology, 2013, 37(4):
 *      203-222.
 */

public class Selection<T> implements Iterator<List<T>> {
	private int nTime, maxId;
	private int[] vals;
	private List<T> next;
	private Map<Integer, T> map;

	/**
	 * Produce all named preference profiles (named = true); all AECs (named =
	 * false)
	 **/
	private boolean named;
	private int watch = -1;

	public Selection(final Collection<? extends T> coll, int nTime, boolean named) {
		if (coll == null) {
			throw new NullPointerException("The collection must not be null");
		}

		map = new HashMap<>();
		int value = 1;
		for (T e : coll)
			map.put(value++, e);

		this.nTime = nTime;
		maxId = coll.size();
		vals = new int[nTime];
		next = new ArrayList<>();
		T e = map.get(1);
		for (int i = 0; i < nTime; i++) {
			next.add(e);
			vals[i] = 1;
		}
		this.named = named;
		this.watch = nTime;
	}

	public boolean hasNext() {
		return next != null;
	}

	public List<T> next() {
		if (!hasNext()) {
			throw new NoSuchElementException();
		}

		boolean end = named ? getNamedNext() : getNext();
		List<T> nextP = null;

		if (!end) {
			nextP = new ArrayList<>();
			for (int k = 0; k < nTime; k++)
				nextP.add(map.get(vals[k]));
		}

		final List<T> result = next;
		next = nextP;
		return result;
	}

	/**
	 * For example, AAAA, [AAAB], [[AAAC]], [AABA], AABB, AABC, [[AACA]],...,
	 * where the strings in square and double-square indicate the redundant
	 * pattern if order is not considered
	 * 
	 * @return
	 */
	boolean getNamedNext() {
		boolean raise = true;
		int i = nTime - 1;
		while (raise && i >= 0) {
			if (vals[i] == maxId) {
				vals[i] = 1;
			} else {
				vals[i]++;
				raise = false;
			}
			i--;
		}
		return raise;
	}

	/**
	 * For example, AAA[A], AAA[B], AA[A]C, AAB[B], AA[B]C, A[A]CC, A[B]CC, ...,
	 * where letters in square indicate the places in change
	 * 
	 * @return
	 */
	boolean getNext() {
		boolean raise = true;
		int i = watch - 1;
		while (raise && i >= 0) {
			if (vals[i] < maxId) {
				vals[i]++;
				if (vals[i] == maxId)
					watch--;
				else {
					// broadcast to right elements
					for (int k = i; k < nTime; k++)
						vals[k] = vals[i];
					watch = nTime;
				}
				raise = false;
			}
			i--;
		}
		return raise;
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		List<String> items = Arrays.asList("A", "B", "C");
		Selection<String> iter = new Selection<>(items, 15, true);

		while (iter.hasNext())
			System.out.println(iter.next());

		TickClock.stopTick();
	}
}
