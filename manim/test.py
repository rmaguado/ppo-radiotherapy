from manim import *
import numpy as np
import random

np.random.seed(0)
random.seed(0)


def get_rand_vector(color=BLACK):
    num_items = 4
    X = np.random.rand(num_items) - 0.5
    vector_items = [Text(f"{x:.5f}", color=color) for x in X[:num_items]]
    vertical_ellipsis = Text("...", color=color)
    vector_items.append(vertical_ellipsis)
    vector_items.append(Text(f"{X[-1]:.5f}", color=color))
    return VGroup(
        *vector_items,
    ).arrange(DOWN, buff=0.3)


def get_distribution(color=BLACK, num_bars=6):
    heights = np.random.rand(num_bars)
    bars = [
        Rectangle(
            width=0.1, height=height, color=color, fill_opacity=1.0, stroke_width=0
        )
        for height in heights
    ]
    bar_group = VGroup(*bars).arrange(RIGHT, buff=0.2)

    for bar, height in zip(bar_group, heights):
        bar.shift(UP * height / 2)

    return bar_group, heights


class PolicyGradientAnimation(Scene):
    def construct(self):
        label_font_size = 32
        self.camera.background_color = WHITE

        environment_img = (
            ImageMobject("environment.png").scale(0.3).to_edge(LEFT, buff=1.5)
        )
        agent_img = ImageMobject("agent.png").scale(0.3).to_edge(RIGHT, buff=3.2)
        thinking_img = ImageMobject("thinking.png").scale(0.4).to_edge(RIGHT, buff=3.2)
        self.add(environment_img, agent_img)
        thinking_img.shift(DOWN * 1.0)
        agent_img.shift(DOWN * 1.0)
        network_position = self.add_neural_network(agent_img)

        # env + agent label
        environment_label = Text("Environment", color=BLACK).next_to(
            environment_img, DOWN
        )
        agent_label = Text("Agent", color=BLACK).next_to(agent_img, DOWN)
        self.add(environment_label, agent_label)

        # observation
        observation_vector = get_rand_vector().scale(0.5)
        observation_vector.move_to(environment_img.get_center())
        self.play(
            FadeIn(observation_vector), observation_vector.animate.move_to(ORIGIN)
        )
        observation_label = Text(
            "Observation", color=BLACK, font_size=label_font_size
        ).next_to(observation_vector, DOWN)
        self.play(FadeIn(observation_label))
        self.play(FadeOut(observation_label))
        self.play(observation_vector.animate.move_to(network_position))
        self.play(FadeOut(observation_vector))

        self.play(FadeOut(agent_img), FadeIn(thinking_img))
        self.wait()
        self.play(FadeOut(thinking_img), FadeIn(agent_img))

        # distribution
        action_probabilities, heights = get_distribution()
        action_probabilities.move_to(network_position)
        self.play(
            FadeIn(action_probabilities), action_probabilities.animate.move_to(ORIGIN)
        )
        distribution_label = Text(
            "Action Probabilities", color=BLACK, font_size=label_font_size
        ).next_to(action_probabilities, DOWN)
        self.play(FadeIn(distribution_label))
        self.play(FadeOut(distribution_label))
        self.play(action_probabilities.animate.shift(UP * 1.5))

        # sample
        bars = action_probabilities
        chosen_index = np.argmax(heights)
        self.highlight_bar(bars, chosen_index, opacity=0.2)

        chosen_bar = bars[chosen_index]

        sampled_action = Text(
            f"a{chosen_index+1}", color=BLACK, font_size=label_font_size
        )
        sampled_action.next_to(chosen_bar, DOWN)
        self.play(FadeIn(sampled_action))
        self.wait()
        self.highlight_bar(bars, chosen_index, opacity=1.0)

        self.play(sampled_action.animate.move_to(environment_img.get_center()))

        self.play(FadeOut(sampled_action))
        self.wait()

        # reward
        reward_text = Text("+0.2", color=GREEN)
        reward_text.move_to(environment_img.get_center())
        self.play(FadeIn(reward_text))
        self.play(reward_text.animate.move_to(ORIGIN))
        reward_label = Text("Reward", color=BLACK, font_size=label_font_size).next_to(
            reward_text, DOWN
        )
        self.play(FadeIn(reward_label))
        self.play(FadeOut(reward_label))
        self.play(reward_text.animate.move_to(action_probabilities.get_center()))
        self.play(FadeOut(reward_text))

        # copy action probabilities
        action_probabilities_copy = action_probabilities.copy()
        self.play(action_probabilities_copy.animate.shift(DOWN * 3))

        # red bar
        chosen_bar_copy = action_probabilities_copy[chosen_index]
        old_height = heights[chosen_index]
        new_height = old_height * 1.2
        height_diff = new_height - heights[chosen_index]
        red_bar = Rectangle(
            width=0.1, height=height_diff, color=RED, fill_opacity=1.0, stroke_width=0
        ).move_to(chosen_bar_copy.get_top() + UP * height_diff / 2)
        modified_probability_label = Text(
            "Modified Probability", color=RED, font_size=label_font_size
        ).next_to(action_probabilities_copy, DOWN)
        self.play(FadeIn(red_bar, modified_probability_label))
        self.play(FadeOut(modified_probability_label))

        self.play(
            FadeOut(action_probabilities, action_probabilities_copy),
            red_bar.animate.move_to(network_position),
        )
        self.play(FadeOut(red_bar))

        # updating weights label
        updating_weights_label = Text("Update Weights", color=BLACK, font_size=28)
        # position it under the network
        updating_weights_label.next_to(agent_img, UP)
        self.play(FadeIn(updating_weights_label))

        # Animate weight updates
        self.animate_network()

        self.play(FadeOut(updating_weights_label))

    def highlight_bar(self, bars, chosen_index, opacity=0.2):
        self.play(
            *[
                bar.animate.set_fill(opacity=opacity)
                for i, bar in enumerate(bars)
                if i != chosen_index
            ],
            run_time=0.5,
        )

    def add_neural_network(self, agent_img):
        layer_nodes = [3, 5, 3]
        layer_spacing = 0.7
        node_spacing = 0.4
        line_thickness = 5

        # Center neural network above the agent
        vertical_offset = UP * 2.2
        horizontal_offset = -(len(layer_nodes) - 1) * layer_spacing / 2

        self.network_lines = []
        for i in range(len(layer_nodes)):
            layer_positions = [
                agent_img.get_center()
                + RIGHT * (i * layer_spacing + horizontal_offset)
                + vertical_offset
                + DOWN * (j - layer_nodes[i] / 2 + 0.5) * node_spacing
                for j in range(layer_nodes[i])
            ]
            if i > 0:
                connections = [
                    Line(
                        left_pos,
                        right_pos,
                        color=GRAY,
                        stroke_width=line_thickness,
                    )
                    for left_pos in prev_layer_positions
                    for right_pos in layer_positions
                ]
                self.add(*connections)
                self.network_lines.extend(connections)
            prev_layer_positions = layer_positions

        return agent_img.get_center() + vertical_offset

    def animate_network(self):
        for _ in range(10):
            line = random.choice(self.network_lines)
            update_color = RED if random.random() > 0.5 else BLUE
            self.play(line.animate.set_color(update_color), run_time=0.05)
            self.play(line.animate.set_color(GRAY), run_time=0.05)
