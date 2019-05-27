from jericho import FrotzEnv


class ZorkEnv:
    def __init__(self, task='egg', verbose=False, word_tokenizer=lambda text: default_tokenizer(text),
                 output_parser=lambda text: text, game_location='gym/zork1.z5', max_steps=20, success_reward=0,
                 pomdp_mode=True, sparse_reward=True):
        self.verbose = verbose

        self.env = FrotzEnv(game_location, seed=0)

        self.prev_s = ''
        self.prev_r = 0
        self.task = task
        self.max_deaths = 0
        self.max_steps = max_steps
        self.steps = 0
        self.success_reward = success_reward
        self.pomdp_mode = pomdp_mode
        self.sparse_reward = sparse_reward

        if self.task == 'egg':
            self.terminal_string = ['You have neither the tools nor the expertise.']
            self.required_items = ['jewel-encrusted egg']
        elif self.task == 'troll':
            self.terminal_string = ['carcass has disappeared', 'Your sword is no longer glowing.']
            self.required_items = []
        elif self.task == 'egg_troll':
            self.terminal_string = ['carcass has disappeared', 'Your sword is no longer glowing.']
            self.required_items = ['jewel-encrusted egg']
        elif self.task == 'full':
            self.terminal_string = ['you win']
            self.required_items = []
        else:
            print('Unknown task: ' + str(self.task))
            raise NotImplementedError

        self.word_tokenizer = word_tokenizer
        self.output_parser = output_parser

    def step(self, action):
        try:
            action_set = set(self.word_tokenizer(action))

            if 'A nasty-looking troll' in self.prev_s and action is not set(['kill', 'troll', 'with', 'sword']) != action_set:
                obs = self.prev_s
                r = 0
                done = False
            else:
                obs, r, done, _ = self.env.step(action)
            self.steps += 1

            reward = -1
            has_won = False

            inventory = self.get_inventory()

            terminal_string_ok = True
            for terminal_string in self.terminal_string:
                if terminal_string not in obs:
                    terminal_string_ok = False
            if terminal_string_ok or self.env.victory():
                missing_items = False
                for item in self.required_items:
                    if item not in inventory:
                        missing_items = True
                if not missing_items:
                    done = True
                    reward = self.success_reward
                    has_won = True

            if self.task == 'full' or not self.sparse_reward:
                reward = reward + r - self.prev_r
                self.prev_r = r
                if has_won:
                    reward = 20

            if not self.pomdp_mode and ('troll' not in obs or 'You can\'t see any troll here' in obs) \
                    and action_set != set(['light', 'match']) \
                    and 'Frigid River' not in obs \
                    and not(action_set == set(['get', 'sceptre']) and 'Frigid River' in self.prev_s) \
                    and action_set != set(['get', 'buoy']) \
                    and 'The thief gestures mysteriously' not in obs \
                    and set(self.word_tokenizer(obs)) != set(['cyclops', 'room']) \
                    and 'thief' not in obs:
                obs, _, _, _ = self.env.step('look')
            self.prev_s = obs

            timeout = False
            if self.steps > self.max_steps:  # self.env.nb_deaths > self.max_deaths or
                done = True
                timeout = True

            return self.parse_game_state(obs, inventory), reward, done, has_won, timeout
        except:
            raise EnvironmentError('There was some error with the Zork env.')

    def get_inventory(self):
        inventory = []
        for item in self.env.get_inventory():
            inventory.append(item.name)
        return inventory

    def reset(self, seed=52):
        self.steps = 0
        self.env.seed = seed
        self.prev_r = 0
        obs = self.env.reset()

        obs = obs.split('840726')[1]

        self.prev_s = obs

        return self.parse_game_state(obs, self.get_inventory())

    def parse_game_state(self, obs, inventory):
        inv = []
        for item in inventory:
            for token in self.word_tokenizer(item):
                inv.append(token)
        return self.output_parser([self.word_tokenizer(obs), inv])

    def close(self):
        self.env.close()

    def render(self):
        #  self.env.render()
        print(self.prev_s)
        print(self.get_inventory())


def default_tokenizer(text):
    words = text.split()
    # remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table).lower() for w in words]
    return stripped
