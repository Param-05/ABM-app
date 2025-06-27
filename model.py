import random
import mesa
from mesa.datacollection import DataCollector

# ————————————————
# Model constants
# ————————————————
ROLE_PROFILES = {
    1: {"tech": 0.9, "soft_skills": 0.1},
    2: {"tech": 0.5, "management": 0.3, "soft_skills": 0.2},
    3: {"management": 0.5, "compliance": 0.3, "soft_skills": 0.2},
    4: {"management": 0.7, "compliance": 0.1, "soft_skills": 0.2},
    5: {"management": 0.8, "soft_skills": 0.1, "compliance": 0.1},
}

BASE_YEARS = {
    1: (0, 3),
    2: (2, 5),
    3: (4, 7),
    4: (6, 10),
    5: (8, 12),
}

LEVEL_DISTRIBUTION = {
    1: 0.4,   # 40% of agents
    2: 0.25,  # 25%
    3: 0.2,   # 20%
    4: 0.1,   # 10%
    5: 0.05,  # 5%
}

LEVEL_EXIT_RATES = {
    1: 0.05,
    2: 0.02,
    3: 0.01,
    4: 0.005,
    5: 0.002,
}

DELTA = {
    1: 0.6,
    2: 0.25,
    3: 0.2,
    4: 0.15,
    5: 0.1,
}


# ————————————————
# Agent
# ————————————————
class WorkerAgent(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.competence = {
            "tech": random.uniform(0, 1),
            "management": random.uniform(0, 1),
            "soft_skills": random.uniform(0, 1),
            "compliance": random.uniform(0, 1),
        }
        self.learning_rate = {
            skill: self.calculate_learning_rate(self.competence[skill])
            for skill in ("tech", "management")
        }

    def calculate_learning_rate(self, C, growth_rate=1):
        return growth_rate * C * (1 - C)

    def set_level(self, level):
        self.level = level

    def set_years_in_company(self, years):
        self.years_in_company = years

    def total_competence(self):
        return sum(self.competence.values())

    def calculate_performance(self):
        profile = ROLE_PROFILES[self.level]
        base = sum(profile[k] * self.competence[k] for k in profile)
        self.performance = max(0, min(1, base))

    def increment_years(self):
        self.years_in_company += 1

    def step(self):
        # you could update competence, learning, etc. here
        pass


def random_tenure(level, base_years=BASE_YEARS, variation=5):
    low, high = base_years[level]
    base = random.randint(low, high)
    jitter = random.randint(-variation, variation)
    return max(0, base + jitter)


# ————————————————
# Model
# ————————————————
# Helper function for assigning random tenure
def random_tenure(level, base_years=BASE_YEARS, variation=5):
        """
        Draw a random tenure for `level`:
        – Start with a uniform pick in BASE_YEARS[level]
        – Add a uniform jitter in [-variation, +variation]
        – Clamp at zero so you never get negative years.
        """
        low, high = base_years[level]
        base   = random.randint(low, high)
        jitter = random.randint(-variation, variation)
        return max(0, base + jitter)

class Organization(mesa.Model):
    def __init__(self, num_agents, promotion_strategy, seed=None):
        super().__init__(seed=seed)
        self.num_agents = num_agents
        self.promotion_strategy = promotion_strategy

        self.last_attritions_by_level = {lvl: [] for lvl in ROLE_PROFILES}
        self.last_promotions_by_level = {lvl: [] for lvl in ROLE_PROFILES}
        
        # self.datacollector = mesa.DataCollector(
        #     agent_reporters={"Total Competence": "total_competence"}
        # )
        # 1) Compute integer quotas
        self.max_slots = {
            lvl: int(LEVEL_DISTRIBUTION[lvl] * num_agents)
            for lvl in LEVEL_DISTRIBUTION
        }
        # slot tracker
        slots_used = {lvl: 0 for lvl in LEVEL_DISTRIBUTION}

        # Create agents
        for _ in range(num_agents):
            a = WorkerAgent(self)
            # Assign level based on distribution
            for lvl in sorted(ROLE_PROFILES.keys(), reverse=True):
                profile = ROLE_PROFILES[lvl]
                # do we still have room?
                if slots_used[lvl] >= self.max_slots[lvl]:
                    continue
                # does agent meet each skill threshold?
                qualifies = all(
                    a.competence[skill] >= profile[skill]
                    for skill in profile
                )
                if qualifies:
                    a.set_level(lvl)
                    a.set_years_in_company(random_tenure(lvl))
                    slots_used[lvl] += 1
                    break
            else:
                # (c) if no break, assign to level 1
                a.set_level(1)
                a.set_years_in_company(random_tenure(lvl))
                slots_used[1] += 1
                pass
            pass

        model_reporters = {
            "Efficiency": lambda m: m.get_efficiency(),
            "Attrited by Level":    lambda m: m.last_attritions_by_level,
            "Promoted by Level":    lambda m: m.last_promotions_by_level,
        }

        agent_reporters = {
            "Level":      lambda a: a.level,
            "Performance":lambda a: a.performance,
            "Years in Company": lambda a: a.years_in_company,
        }
        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters
        )

        pass


    def merit_based_promotion(self, candidates, threshold=0.8):
         # 1) Filter by threshold
        qualified = [a for a in candidates if a.performance >= threshold]
        # 2) Sort descending by performance
        qualified_sorted = sorted(
            qualified,
            key=lambda a: a.performance,)
        return qualified_sorted

    def seniority_based_promotion(self, candidates, years=5):
        # 1) Filter by threshold
        qualified = [a for a in candidates if a.years_in_company >= years]
        # 2) Sort descending by years_in_company
        qualified_sorted = sorted(
            qualified,
            key=lambda a: a.years_in_company,)
        return qualified_sorted

    def hybrid_promotion(self, candidates, perf_weight=0.5, seniority_weight=0.5):
        # 1) Figure out max possible tenure for normalization
        max_years = max(end for (_, end) in BASE_YEARS.values())
        # 2) Calculate weighted scores
        qualified_sorted = sorted(
        candidates,
        key=lambda a: (
            perf_weight * a.performance
            + seniority_weight * (a.years_in_company / max_years)
        ))
        return qualified_sorted

    def random_promotion(self, candidates):
        """Shuffle all candidates, so popping returns a random one."""
        shuffled = list(candidates)
        random.shuffle(shuffled)
        return shuffled
    
    # Touranment is not implemented yet because it is similar to the merit-based

    # Implementing attrition
    # def apply_attrition(self):
    #     to_remove = []
    #     for lvl, rate in LEVEL_EXIT_RATES.items():
    #         # select all agents at this level
    #         lvl_agents = list(self.agents.select(lambda a: a.level == lvl))
    #         n = int(rate * len(lvl_agents))
    #         to_remove += random.sample(lvl_agents, n)
    #     for a in to_remove:
    #         a.remove()   # removes from self.agents

    def apply_attrition(self):
        # reset
        self.last_attritions_by_level = {lvl: [] for lvl in ROLE_PROFILES}
        to_remove = []
        for lvl, rate in LEVEL_EXIT_RATES.items():
            lvl_agents = list(self.agents.select(lambda a: a.level == lvl))
            n = int(rate * len(lvl_agents))
            removed = self.random.sample(lvl_agents, n)
            self.last_attritions_by_level[lvl] = [a.unique_id for a in removed]
            to_remove.extend(removed)
        for a in to_remove:
            a.remove()
    
    # def promote_agents(self):
    #     top = max(ROLE_PROFILES)
    #     for lvl in range(1, top):
    #         # vacancies = capacity minus current headcount
    #         occupied = len(self.agents.select(lambda a: a.level == lvl+1))
    #         vacancies = self.max_slots[lvl+1] - occupied
    #         if vacancies <= 0:
    #             continue

    #         candidates = list(self.agents.select(lambda a: a.level == lvl))
    #         if not candidates:
    #             continue

    #         strat = self.promotion_strategy
    #         if strat == "merit":
    #             stack = self.merit_based_promotion(candidates)
    #         elif strat == "seniority":
    #             stack = self.seniority_based_promotion(candidates)
    #         elif strat == "hybrid":
    #             stack = self.hybrid_promotion(candidates)
    #         elif strat == "random":
    #             stack = self.random_promotion(candidates)
    #         elif strat == "tournament":
    #             stack = self.tournament_promotion(candidates)
    #         else:
    #             raise ValueError(f"Unknown strategy {strat}")

    #         # promote by popping off the best until slots fill
    #         while vacancies and stack:
    #             p = stack.pop()
    #             p.set_level(lvl+1)
    #             p.years_in_company = 0
    #             p.calculate_performance()
    #             vacancies -= 1
    def promote_agents(self):
        # reset
        self.last_promotions_by_level = {lvl: [] for lvl in ROLE_PROFILES}

        top = max(ROLE_PROFILES)
        for lvl in range(1, top):
            occupied  = len(self.agents.select(lambda a: a.level == lvl+1))
            vacancies = self.max_slots[lvl+1] - occupied
            if vacancies <= 0: 
                continue

            candidates = list(self.agents.select(lambda a: a.level == lvl))
            # pick stack according to strategy
            if self.promotion_strategy == "merit":
                stack = self.merit_based_promotion(candidates)
            elif self.promotion_strategy == "seniority":
                stack = self.seniority_based_promotion(candidates)
            # … other strategies …

            while vacancies and stack:
                p = stack.pop()
                p.promoted_this_step = True
                p.set_level(lvl+1)
                p.years_in_company = 0
                p.calculate_performance()
                self.last_promotions_by_level[lvl].append(p.unique_id)
                vacancies -= 1
    
    def get_efficiency(self):
        """
        Returns the average performance across all agents,
        a number in [0,1].
        """
        self.agents.do('calculate_performance')
        all_agents = list(self.agents)
        if not all_agents:
            return 0
        total = sum(a.performance for a in all_agents)
        return total / len(all_agents)
    


    def hire_new_level1(self):
        """Fill any vacancies at Level 1 with brand‐new agents."""
        # 1) Count how many are currently in Level 1
        current_L1 = len(self.agents.select(lambda a: a.level == 1))
        # 2) How many slots is Level 1 allowed?
        cap_L1 = self.max_slots[1]
        vacancies = cap_L1 - current_L1
        # 3) Create that many new agents
        for _ in range(vacancies):
            new = WorkerAgent(self)   # Mesa 3.x: auto‐assigns unique_id + adds to self.agents
            new.set_level(1)
            new.years_in_company = 0
            new.calculate_performance()

        
    def step(self):
        self.agents.do('increment_years')
        self.agents.do('calculate_performance')
        self.apply_attrition()
        self.promote_agents()
        self.hire_new_level1()
        self.datacollector.collect(self)
