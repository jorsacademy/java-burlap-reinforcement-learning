package com.jorsacademy.burlap;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.singleagent.SADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/** Runs the learning benchmark, compares it with Value Iteration, and exports CSV/SVG results. */
public final class BurlapExperimentComparison {

    private static final Path RESULTS_DIR = Path.of("results");
    private static final Path CSV_PATH = RESULTS_DIR.resolve("learning-curve.csv");
    private static final Path SVG_PATH = RESULTS_DIR.resolve("learning-curve.svg");

    private BurlapExperimentComparison() {}

    public static void main(String[] args) throws IOException {
        BurlapQLearningBenchmark.TrainingResult q = BurlapQLearningBenchmark.train(
                BurlapQLearningBenchmark.TRAINING_EPISODES,
                BurlapQLearningBenchmark.MAX_STEPS_PER_EPISODE);
        int viSteps = valueIterationSteps();

        Files.createDirectories(RESULTS_DIR);
        writeCsv(q.episodeSteps());
        writeSvg(q.episodeSteps(), viSteps);

        System.out.println("BURLAP experiment: Q-Learning vs Value Iteration");
        System.out.printf(Locale.US, "Q-learning first-50 average : %.2f%n", q.firstWindowAverage());
        System.out.printf(Locale.US, "Q-learning last-50 average  : %.2f%n", q.lastWindowAverage());
        System.out.printf(Locale.US, "Q-learning improvement      : %.2f%%%n", q.improvementPercent());
        System.out.println("Q-learning best episode      : " + q.bestEpisodeSteps());
        System.out.println("Value Iteration rollout      : " + viSteps + " actions");
        System.out.println("CSV                           : " + CSV_PATH);
        System.out.println("Learning curve                : " + SVG_PATH);
    }

    public static int valueIterationSteps() {
        GridWorldDomain gridWorld = new GridWorldDomain(
                BurlapQLearningBenchmark.GRID_SIZE,
                BurlapQLearningBenchmark.GRID_SIZE);
        gridWorld.setMapToFourRooms();
        gridWorld.setTf(new GridWorldTerminalFunction(
                BurlapQLearningBenchmark.GOAL_X,
                BurlapQLearningBenchmark.GOAL_Y));
        SADomain domain = gridWorld.generateDomain();
        GridWorldState initial = new GridWorldState(0, 0);

        ValueIteration planner = new ValueIteration(
                domain,
                BurlapQLearningBenchmark.DISCOUNT,
                new SimpleHashableStateFactory(),
                1e-5,
                1000);
        planner.toggleReachabiltiyTerminalStatePruning(true);
        GreedyQPolicy policy = planner.planFromState(initial);
        Episode rollout = PolicyUtils.rollout(
                policy,
                initial,
                domain.getModel(),
                BurlapQLearningBenchmark.MAX_STEPS_PER_EPISODE);
        return rollout.numActions();
    }

    static void writeCsv(List<Integer> steps) throws IOException {
        StringBuilder out = new StringBuilder("episode,actions,moving_average_25\n");
        for (int i = 0; i < steps.size(); i++) {
            int from = Math.max(0, i - 24);
            double avg = steps.subList(from, i + 1).stream().mapToInt(Integer::intValue).average().orElse(0.0);
            out.append(i + 1).append(',').append(steps.get(i)).append(',')
                    .append(String.format(Locale.US, "%.3f", avg)).append('\n');
        }
        Files.writeString(CSV_PATH, out.toString(), StandardCharsets.UTF_8);
    }

    static void writeSvg(List<Integer> steps, int viSteps) throws IOException {
        int width = 1000, height = 560;
        int left = 70, right = 25, top = 35, bottom = 65;
        int plotW = width - left - right, plotH = height - top - bottom;
        int maxY = Math.max(50, steps.stream().mapToInt(Integer::intValue).max().orElse(50));

        StringBuilder polyline = new StringBuilder();
        for (int i = 0; i < steps.size(); i++) {
            int from = Math.max(0, i - 24);
            double avg = steps.subList(from, i + 1).stream().mapToInt(Integer::intValue).average().orElse(0.0);
            double x = left + plotW * (i / (double) Math.max(1, steps.size() - 1));
            double y = top + plotH * (1.0 - Math.min(avg, maxY) / maxY);
            polyline.append(String.format(Locale.US, "%.2f,%.2f ", x, y));
        }
        double viY = top + plotH * (1.0 - Math.min(viSteps, maxY) / (double) maxY);

        String svg = """
                <svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560">
                  <rect width="1000" height="560" fill="white"/>
                  <text x="500" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">BURLAP Four Rooms: Q-Learning Convergence</text>
                  <line x1="70" y1="35" x2="70" y2="495" stroke="black"/>
                  <line x1="70" y1="495" x2="975" y2="495" stroke="black"/>
                  <text x="520" y="545" text-anchor="middle" font-family="sans-serif" font-size="14">Training episode</text>
                  <text x="18" y="265" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 18 265)">Actions per episode</text>
                  <polyline points="%s" fill="none" stroke="#1f77b4" stroke-width="2"/>
                  <line x1="70" y1="%.2f" x2="975" y2="%.2f" stroke="#d62728" stroke-width="2" stroke-dasharray="7 5"/>
                  <text x="965" y="%.2f" text-anchor="end" font-family="sans-serif" font-size="13">Value Iteration: %d actions</text>
                  <text x="80" y="55" font-family="sans-serif" font-size="13">Blue: Q-learning 25-episode moving average</text>
                </svg>
                """.formatted(polyline, viY, viY, Math.max(50, viY - 7), viSteps);
        Files.writeString(SVG_PATH, svg, StandardCharsets.UTF_8);
    }
}
