# Student Career Recommendation System (Neo4j)

**Big Data Management (MAXD 5123), Project 1, UTeM · May 2026 · group of 3**
Group 2: Hanis binti Hussin, Muhammad Iman Firdaus bin Md Rostan, Norarmiza binti Arsad.
Instructors: Ts. Dr. Norfadzlia binti Mohd Yusof, Ts. Dr. Abdul Syukor bin Mohamad Jaya.

## Why a graph database
Matching students to jobs is a relationship problem: student -> courses -> skills -> jobs.
In a relational schema every recommendation is a pile of JOINs that gets slower as the data
grows. Neo4j stores the relationships themselves, so a recommendation is a traversal.

## The model
**Nodes:** `Student`, `Course`, `Skill`, `Job`
**Relationships:** `TAKES`, `HAS_SKILL`, `TEACHES`, `REQUIRES`, `RELATED_TO`

## What the queries answer
- Top recommended jobs for a student, by skills already held
- The connection path between a student and a target job
- Skill gaps, and which courses close them
- Students with similar skills or interests (peer and mentor finding)
- A "market readiness" score per student
- Most in-demand skills, and students with no recommendation yet (the ones needing advice)

## Files
```
career_recommender.cypher   all queries in order, ready to paste into Neo4j Browser
queries-export.csv          the original Neo4j Browser favourites export
```

## Run it
Install Neo4j Desktop or run `neo4j` in Docker, open Neo4j Browser, then execute
`career_recommender.cypher` from the top: it creates the database, loads sample students,
courses, skills and jobs, then runs the recommendation queries.

## Figures

The 13 images below are the real figures from the MAXD 5123 project report, extracted from the submitted PDF.

![The graph model: Student, Skill, Job, Course, Mentor and the relationships between them.](figures/01-graph-model-overview.png)

*The graph model: Student, Skill, Job, Course, Mentor and the relationships between them.*

![One student traced through their skills to the jobs those skills unlock.](figures/02-student-skill-job-paths.png)

*One student traced through their skills to the jobs those skills unlock.*

![The full graph after loading every CSV.](figures/03-full-graph-layout.png)

*The full graph after loading every CSV.*

![The MERGE statements — MERGE not CREATE, so re-running never duplicates edges.](figures/04-cypher-merge-relationships.png)

*The MERGE statements — MERGE not CREATE, so re-running never duplicates edges.*

![Database overview: labels, relationship types and counts after loading.](figures/05-database-overview.png)

*Database overview: labels, relationship types and counts after loading.*

![DevOps / Docker / Cloud subgraph: course connects learner to job through a skill.](figures/06-skill-course-subgraph.png)

*DevOps / Docker / Cloud subgraph: course connects learner to job through a skill.*

![The recommendation traversal rendered as a graph.](figures/07-recommendation-graph.png)

*The recommendation traversal rendered as a graph.*

![Learners matched to mentors through shared skills.](figures/08-learner-mentor-graph.png)

*Learners matched to mentors through shared skills.*

![How many students each job was recommended to.](figures/09-job-recommendation-counts.png)

*How many students each job was recommended to.*

![Each course mapped to the jobs it feeds into.](figures/10-course-to-related-jobs.png)

*Each course mapped to the jobs it feeds into.*

![Skill demand: how many students already hold each skill.](figures/11-skill-demand-counts.png)

*Skill demand: how many students already hold each skill.*

![Skill-gap query: target job → missing skill → the course that teaches it.](figures/12-skill-gap-recommendations.png)

*Skill-gap query: target job → missing skill → the course that teaches it.*

![Learner → skill to learn → potential mentors who already have it.](figures/13-mentor-matching.png)

*Learner → skill to learn → potential mentors who already have it.*
