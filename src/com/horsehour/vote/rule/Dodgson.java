package com.horsehour.vote.rule;

/**
 * Dodgson's method is a voting system proposed by Charles Dodgson. Each voter
 * submits an ordered list of all candidates according to their own preference
 * (from best to worst). The winner is defined to be the candidate for whom we
 * need to perform the minimum number of pairwise swaps (added over all
 * candidates) before they become a Condorcet winner. In particular, if there is
 * already a Condorcet winner, they win the election.
 * <p>
 * In short, we must find the voting profile with minimum Kendall tau distance
 * from the input, such that it has a Condorcet winner; they are declared the
 * victor. Computing the winner is an NP-hard problem.
 * <p>
 * Dodgson isn't known for his mathematical work; he's famous as Lewis Carroll
 * for writing Alice's Adventures in Wonderland, its sequel, and other
 * fantasies. Doron Zeilberger of Rutgers University described him as perhaps
 * the most famous mathematician in the world.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 8:06:54 PM, Jul 31, 2016
 *
 */

public class Dodgson {

}
