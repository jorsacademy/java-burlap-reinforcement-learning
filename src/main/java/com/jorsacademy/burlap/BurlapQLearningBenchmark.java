package com.jorsacademy.burlap;

import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.statehashing.simple.SimpleHashableStateFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Compact BURLAP benchmark that trains a tabular Q-learning agent on the
 * classic 11x11 four-rooms navigation problem.
 *
 * <p>The environment uses BURLAP's default step-cost reward model. Therefore,
 * a better policy reaches the terminal state in fewer actions. The program
 * reports the average episode length at the beginning and end of training so
 * that learning progress is directly visible from the command line.</p>
 */
public final class BurlapQLearningBenchmark {

    public static final int GRID_SIZE = 11;
    public static final int GOAL_X = 10;
    public static final int GOAL_Y = 10;
    public static final int TRAINING_EPISODES = 500;
    public static final int MAX_STEPS_PER_EPISODE = 500;
    public static final int SUMMARY_WINDOW = 50;
    public static final double DISCOUNT = 0.99;
    public static final double INITIAL_Q = 1.0;
    public static final double LEARNING_RATE = 0.5;

    private BurlapQLearningBenchmark() {
    }

    public static void main(String[] args) {
        TrainingResult result = train(TRAINING_EPISODES, MAX_STEPS_PER_EPISODE);

        System.out.println("BURLAP Q-learning benchmark");
        System.out.println("Environment : 11x11 Four Rooms");
        System.out.println("Start       : (0, 0)");
        System.out.println("Goal        : (10, 10)");
        System.out.println("Episodes    : " + result.episodeSteps().size());
        System.out.println();
        System.out.printf(Locale.US, "First %d avg actions : %.2f%n",
                SUMMARY_WINDOW, result.firstWindowAverage());
        System.out.printf(Locale.US, "Last  %d avg actions : %.2f%n",
                SUMMARY_WINDOW, result.lastWindowAverage());
        System.out.printf(Locale.US, "Improvement          : %.2f%%%n",
                result.improvementPercent());
        System.out.println("Best episode actions : " + result.bestEpisodeSteps());

        if (result.lastWindowAverage() < result.firstWindowAverage()) {
            System.out.println("Result      : learning improved navigation efficiency.");
        } else {
            System.out.println("Result      : no improvement detected in this stochastic run.");
        }
    }

    /**
     * Trains a fresh Q-learning agent and returns episode-length statistics.
     */
    public static TrainingResult train(int episodes, int maxStepsPerEpisode) {
        if (episodes < 2) {
            throw new IllegalArgumentException("episodes must be at least 2");
        }
        if (maxStepsPerEpisode < 1) {
            throw new IllegalArgumentException("maxStepsPerEpisode must be positive");
        }

        GridWorldDomain gridWorld = new GridWorldDomain(GRID_SIZE, GRID_SIZE);
        gridWorld.setMapToFourRooms();
        gridWorld.setTf(new GridWorldTerminalFunction(GOAL_X, GOAL_Y));

        SADomain domain = gridWorld.generateDomain();
        Environment environment = new SimulatedEnvironment(
                domain,
                new GridWorldState(0, 0)
        );

        QLearning agent = new QLearning(
                domain,
                DISCOUNT,
                new SimpleHashableStateFactory(),
                INITIAL_Q,
                LEARNING_RATE
        );

        List<Integer> steps = new ArrayList<>(episodes);
        for (int episode = 0; episode < episodes; episode++) {
            Episode outcome = agent.runLearningEpisode(environment, maxStepsPerEpisode);
            steps.add(outcome.numActions());
            environment.resetEnvironment();
        }

        int window = Math.min(SUMMARY_WINDOW, Math.max(1, episodes / 2));
        double firstAverage = average(steps, 0, window);
        double lastAverage = average(steps, steps.size() - window, steps.size());
        int best = steps.stream().mapToInt(Integer::intValue).min().orElse(maxStepsPerEpisode);

        return new TrainingResult(List.copyOf(steps), firstAverage, lastAverage, best);
    }

    private static double average(List<Integer> values, int fromInclusive, int toExclusive) {
        return values.subList(fromInclusive, toExclusive)
                .stream()
                .mapToInt(Integer::intValue)
                .average()
                .orElse(Double.NaN);
    }

    public record TrainingResult(
            List<Integer> episodeSteps,
            double firstWindowAverage,
            double lastWindowAverage,
            int bestEpisodeSteps
    ) {
        public double improvementPercent() {
            if (firstWindowAverage == 0.0) {
                return 0.0;
            }
            return 100.0 * (firstWindowAverage - lastWindowAverage) / firstWindowAverage;
        }
    }
}
