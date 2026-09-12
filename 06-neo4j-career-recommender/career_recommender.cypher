// Student Career Recommendation System - Neo4j / Cypher
// MAXD 5123 Big Data Management, Project 1, UTeM
// Group 2: Hanis Binti Hussin, Muhammad Iman Firdaus Bin Md Rostan, Norarmiza Binti Arsad

// ---------------------------------------------------------------
// 1. Create Database
// ---------------------------------------------------------------
// 1. Create Database

CREATE DATABASE careerdb;

// ---------------------------------------------------------------
// 2. Use Database
// ---------------------------------------------------------------
// 2. Use Database

:use careerdb

// ---------------------------------------------------------------
// 3. Delete Existing Nodes and Relationships
// ---------------------------------------------------------------
// 3. Delete Existing Nodes and Relationships

MATCH (n)
DETACH DELETE n;

// ---------------------------------------------------------------
// 4. Create Student Nodes
// ---------------------------------------------------------------
CREATE
(:Student {id:1, name:'Ali', interest:'Data Science'}),
(:Student {id:2, name:'Siti', interest:'Artificial Intelligence'}),
(:Student {id:3, name:'John', interest:'Cybersecurity'}),
(:Student {id:4, name:'Ahmad', interest:'Cloud Computing'}),
(:Student {id:5, name:'Lisa', interest:'Web Development'}),
(:Student {id:6, name:'Daniel', interest:'Data Visualization'}),
(:Student {id:7, name:'Aina', interest:'Machine Learning'}),
(:Student {id:8, name:'Kumar', interest:'Software Engineering'}),
(:Student {id:9, name:'Farah', interest:'Business Intelligence'}),
(:Student {id:10, name:'Michael', interest:'Artificial Intelligence'}),
(:Student {id:11, name:'Sara', interest:'Cybersecurity'}),
(:Student {id:12, name:'Hakim', interest:'Cloud Computing'}),
(:Student {id:13, name:'Emily', interest:'Web Development'}),
(:Student {id:14, name:'Adam', interest:'Machine Learning'}),
(:Student {id:15, name:'Sophia', interest:'Data Analytics'});

// ---------------------------------------------------------------
// 5. Create Course Nodes
// ---------------------------------------------------------------
// 5. Create Course Nodes

CREATE
(:Course {id:1, name:'Python Programming'}),
(:Course {id:2, name:'Machine Learning'}),
(:Course {id:3, name:'Cybersecurity'}),
(:Course {id:4, name:'Cloud Computing'}),
(:Course {id:5, name:'Web Development'}),
(:Course {id:6, name:'Data Visualization'}),
(:Course {id:7, name:'Artificial Intelligence'}),
(:Course {id:8, name:'Software Engineering'}),
(:Course {id:9, name:'Business Intelligence'}),
(:Course {id:10, name:'Database Systems'}),
(:Course {id:11, name:'Networking'}),
(:Course {id:12, name:'Java Programming'}),
(:Course {id:13, name:'Big Data Analytics'}),
(:Course {id:14, name:'DevOps'}),
(:Course {id:15, name:'UI UX Design'});

// ---------------------------------------------------------------
// 6. Create Skill Nodes
// ---------------------------------------------------------------
// 6. Create Skill Nodes

CREATE
(:Skill {id:1, name:'Python'}),
(:Skill {id:2, name:'Machine Learning'}),
(:Skill {id:3, name:'Cybersecurity'}),
(:Skill {id:4, name:'Cloud'}),
(:Skill {id:5, name:'JavaScript'}),
(:Skill {id:6, name:'Visualization'}),
(:Skill {id:7, name:'Artificial Intelligence'}),
(:Skill {id:8, name:'Software Design'}),
(:Skill {id:9, name:'Power BI'}),
(:Skill {id:10, name:'SQL'}),
(:Skill {id:11, name:'Networking'}),
(:Skill {id:12, name:'Java'}),
(:Skill {id:13, name:'Big Data'}),
(:Skill {id:14, name:'Docker'}),
(:Skill {id:15, name:'UI UX'});

// ---------------------------------------------------------------
// 7. Create Job Nodes
// ---------------------------------------------------------------
// 7. Create Job Nodes

CREATE
(:Job {id:1, title:'Data Scientist'}),
(:Job {id:2, title:'AI Engineer'}),
(:Job {id:3, title:'Cybersecurity Analyst'}),
(:Job {id:4, title:'Cloud Engineer'}),
(:Job {id:5, title:'Web Developer'}),
(:Job {id:6, title:'Data Analyst'}),
(:Job {id:7, title:'ML Engineer'}),
(:Job {id:8, title:'Software Engineer'}),
(:Job {id:9, title:'BI Analyst'}),
(:Job {id:10, title:'Database Administrator'}),
(:Job {id:11, title:'Network Engineer'}),
(:Job {id:12, title:'Java Developer'}),
(:Job {id:13, title:'Big Data Engineer'}),
(:Job {id:14, title:'DevOps Engineer'}),
(:Job {id:15, title:'UI UX Designer'});

// ---------------------------------------------------------------
// 8. Create Student Takes Course Relationships
// ---------------------------------------------------------------
// 8. Create Student Takes Course Relationships

UNWIND RANGE(1,15) AS i

MATCH (s:Student {id:i}),
      (c:Course {id:i})

CREATE (s)-[:TAKES]->(c);

// ---------------------------------------------------------------
// 9. Create Student Has Skill Relationships
// ---------------------------------------------------------------
// 9. Create Student Has Skill Relationships

UNWIND RANGE(1,15) AS i

MATCH (s:Student {id:i}),
      (sk:Skill {id:i})

CREATE (s)-[:HAS_SKILL]->(sk);

// ---------------------------------------------------------------
// 10. Additional skills for students
// ---------------------------------------------------------------
// Additional skills for students

MATCH (s:Student {id:1}), (sk:Skill {id:2})
CREATE (s)-[:HAS_SKILL]->(sk);

MATCH (s:Student {id:2}), (sk:Skill {id:7})
CREATE (s)-[:HAS_SKILL]->(sk);

MATCH (s:Student {id:3}), (sk:Skill {id:11})
CREATE (s)-[:HAS_SKILL]->(sk);

MATCH (s:Student {id:4}), (sk:Skill {id:14})
CREATE (s)-[:HAS_SKILL]->(sk);

MATCH (s:Student {id:5}), (sk:Skill {id:5})
CREATE (s)-[:HAS_SKILL]->(sk);

// ---------------------------------------------------------------
// Display Takes
// ---------------------------------------------------------------
MATCH p=()-[:TAKES]->() RETURN p LIMIT 25;

// ---------------------------------------------------------------
// Display Has Skill
// ---------------------------------------------------------------
MATCH p=()-[:HAS_SKILL]->() RETURN p LIMIT 25;

// ---------------------------------------------------------------
// 10. Create Course Teaches Skill Relationships
// ---------------------------------------------------------------
// 10. Create Course Teaches Skill Relationships

UNWIND RANGE(1,15) AS i

MATCH (c:Course {id:i}),
      (sk:Skill {id:i})

CREATE (c)-[:TEACHES]->(sk);

// ---------------------------------------------------------------
// 11. Create Job Requires Skill Relationships
// ---------------------------------------------------------------
// 11. Create Job Requires Skill Relationships

UNWIND RANGE(1,15) AS i

MATCH (j:Job {id:i}),
      (sk:Skill {id:i})

CREATE (j)-[:REQUIRES]->(sk);

// ---------------------------------------------------------------
// Additional required skills for jobs
// ---------------------------------------------------------------
// Additional required skills for jobs

MATCH (j:Job {id:1}), (sk:Skill {id:2})
CREATE (j)-[:REQUIRES]->(sk);

MATCH (j:Job {id:2}), (sk:Skill {id:7})
CREATE (j)-[:REQUIRES]->(sk);

MATCH (j:Job {id:3}), (sk:Skill {id:11})
CREATE (j)-[:REQUIRES]->(sk);

MATCH (j:Job {id:4}), (sk:Skill {id:14})
CREATE (j)-[:REQUIRES]->(sk);

MATCH (j:Job {id:5}), (sk:Skill {id:5})
CREATE (j)-[:REQUIRES]->(sk);

// ---------------------------------------------------------------
// 12. Create Job Related To Course Relationships
// ---------------------------------------------------------------
// 12. Create Job Related To Course Relationships

UNWIND RANGE(1,15) AS i

MATCH (j:Job {id:i}),
      (c:Course {id:i})

CREATE (j)-[:RELATED_TO]->(c);

// ---------------------------------------------------------------
// 13. Display All Nodes and Relationships
// ---------------------------------------------------------------
// 13. Display All Nodes and Relationships

MATCH p=(n)-[r]->(m)
RETURN p;

// ---------------------------------------------------------------
// Remove duplicate HAS_SKILL relationships for Lisa
// ---------------------------------------------------------------
// Remove duplicate HAS_SKILL relationships for Lisa

MATCH (s:Student {name:'Lisa'})-[r:HAS_SKILL]->(sk:Skill {name:'JavaScript'})
WITH collect(r) AS rels
FOREACH (r IN tail(rels) | DELETE r);

// ---------------------------------------------------------------
// Remove duplicate relationships between
// ---------------------------------------------------------------
// Remove duplicate relationships between
// Web Developer and JavaScript

MATCH (j:Job {title:'Web Developer'})-[r:REQUIRES]->(sk:Skill {name:'JavaScript'})

WITH collect(r) AS rels

FOREACH (r IN tail(rels) | DELETE r);

// ---------------------------------------------------------------
// Display Student Profile with Recommended Jobs
// ---------------------------------------------------------------
// Display Student Profile with Recommended Jobs

MATCH (s:Student)

OPTIONAL MATCH (s)-[:TAKES]->(c:Course)

OPTIONAL MATCH (s)-[:HAS_SKILL]->(sk:Skill)

OPTIONAL MATCH (s)-[:HAS_SKILL]->(:Skill)<-[:REQUIRES]-(j1:Job)

OPTIONAL MATCH (s)-[:TAKES]->(:Course)<-[:RELATED_TO]-(j2:Job)

WITH
s,
collect(DISTINCT c.name) AS Courses,
collect(DISTINCT sk.name) AS Skills,

collect(DISTINCT j1.title) +
collect(DISTINCT j2.title) AS AllJobs

RETURN
s.name AS Student,
s.interest AS Interest,
Courses,
Skills,

reduce(uniqueJobs = [], job IN AllJobs |
    CASE
        WHEN job IN uniqueJobs THEN uniqueJobs
        ELSE uniqueJobs + job
    END
) AS RecommendedJobs;

// ---------------------------------------------------------------
// Find students with similar skills
// ---------------------------------------------------------------
// Find students with similar skills

MATCH (s1:Student)-[:HAS_SKILL]->(sk:Skill)<-[:HAS_SKILL]-(s2:Student)

WHERE s1 <> s2

RETURN
s1.name AS Student1,
s2.name AS Student2,

collect(DISTINCT sk.name) AS CommonSkills;

// ---------------------------------------------------------------
// 17. Find Connection Path Between Student and Job
// ---------------------------------------------------------------
// 17. Find Connection Path Between Student and Job

// This query shows the relationship path
// between a student and a recommended job

MATCH path =

(s:Student {name:'Ali'})-[*]-(j:Job {title:'Data Scientist'})

RETURN path;

// ---------------------------------------------------------------
// 18. Find Top Recommended Jobs
// ---------------------------------------------------------------
// 18. Find Top Recommended Jobs

// This query counts how many students
// are recommended for each job

MATCH (s:Student)

OPTIONAL MATCH
(s)-[:HAS_SKILL]->(:Skill)<-[:REQUIRES]-(j1:Job)

OPTIONAL MATCH
(s)-[:TAKES]->(:Course)<-[:RELATED_TO]-(j2:Job)

WITH
s,
collect(DISTINCT j1.title) +
collect(DISTINCT j2.title) AS AllJobs

UNWIND AllJobs AS JobName

RETURN
JobName AS Job,
COUNT(DISTINCT s) AS TotalRecommendedStudents

ORDER BY TotalRecommendedStudents DESC;

// ---------------------------------------------------------------
// 19. Find Skills Required By Each Job
// ---------------------------------------------------------------
// 19. Find Skills Required By Each Job

// This query shows what skills
// are required for every job

MATCH (j:Job)-[:REQUIRES]->(sk:Skill)

RETURN
j.title AS Job,
collect(DISTINCT sk.name) AS RequiredSkills

ORDER BY Job;

// ---------------------------------------------------------------
// 20. Find Courses That Help Students Get Jobs
// ---------------------------------------------------------------
// 20. Find Courses That Help Students Get Jobs

// This query shows which courses
// are related to which jobs

MATCH (c:Course)<-[:RELATED_TO]-(j:Job)

RETURN
c.name AS Course,
collect(DISTINCT j.title) AS RelatedJobs

ORDER BY Course;

// ---------------------------------------------------------------
// 21. Find Students Without Job Recommendations
// ---------------------------------------------------------------
// 21. Find Students Without Job Recommendations

// This query checks if any student
// has no matching job recommendation

MATCH (s:Student)

WHERE NOT EXISTS {

    MATCH (s)-[:HAS_SKILL]->(:Skill)<-[:REQUIRES]-(:Job)

}

AND NOT EXISTS {

    MATCH (s)-[:TAKES]->(:Course)<-[:RELATED_TO]-(:Job)

}

RETURN
s.name AS Student,
s.interest AS Interest;

// ---------------------------------------------------------------
// 22. Count Total Nodes By Category
// ---------------------------------------------------------------
// 22. Count Total Nodes By Category

// This query counts total records
// for each node type

MATCH (s:Student)
WITH COUNT(s) AS TotalStudents

MATCH (c:Course)
WITH TotalStudents, COUNT(c) AS TotalCourses

MATCH (sk:Skill)
WITH TotalStudents, TotalCourses, COUNT(sk) AS TotalSkills

MATCH (j:Job)

RETURN
TotalStudents,
TotalCourses,
TotalSkills,
COUNT(j) AS TotalJobs;

// ---------------------------------------------------------------
// 23. Display Complete Student Career Recommendation Graph
// ---------------------------------------------------------------
// 23. Display Complete Student Career Recommendation Graph

// This query displays the complete graph
// with all nodes and relationships

MATCH p=(n)-[r]->(m)

RETURN p;

// ---------------------------------------------------------------
// 24. Find Most Popular Skills Among Students
// ---------------------------------------------------------------
// 24. Find Most Popular Skills Among Students

// This query counts how many students
// have each skill

MATCH (s:Student)-[:HAS_SKILL]->(sk:Skill)

RETURN
sk.name AS Skill,
COUNT(DISTINCT s) AS TotalStudents

ORDER BY TotalStudents DESC;

// ---------------------------------------------------------------
// 25. Find Most Demanded Skills By Jobs
// ---------------------------------------------------------------
// 25. Find Most Demanded Skills By Jobs

// This query counts how many jobs
// require each skill

MATCH (j:Job)-[:REQUIRES]->(sk:Skill)

RETURN
sk.name AS Skill,
COUNT(DISTINCT j) AS TotalJobsRequiringSkill

ORDER BY TotalJobsRequiringSkill DESC;

// ---------------------------------------------------------------
// 26. Find Students Qualified For Specific Job (Data Scientist)
// ---------------------------------------------------------------
// 26. Find Students Qualified For Specific Job (Data Scientist)

// This query finds students
// who match a selected job

MATCH (s:Student)-[:HAS_SKILL]->(sk:Skill)<-[:REQUIRES]-
(j:Job {title:'Data Scientist'})

RETURN
j.title AS Job,
collect(DISTINCT s.name) AS QualifiedStudents,
collect(DISTINCT sk.name) AS MatchingSkills;
