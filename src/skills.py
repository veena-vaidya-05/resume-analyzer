"""Skill database and skill extraction logic.

Requirement: predefined skill database with 300+ technical skills.
This file implements a categorized skill list plus robust extraction.

Extraction logic:
- build a keyword/phrase matcher from the database
- match against resume and job description text
- compute matching/missing/extra skills
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple


@dataclass(frozen=True)
class SkillMatch:
    matching_skills: List[str]
    missing_skills: List[str]
    extra_skills: List[str]
    # category -> list
    by_category: Dict[str, List[str]]


def _normalize_phrase(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def build_skill_database() -> Dict[str, List[str]]:
    """Return categories -> list of skills/keywords (>= 300 total)."""

    # The skills list is intentionally extensive to satisfy the 300+ requirement.
    # Use lowercase keywords for matching convenience.

    db: Dict[str, List[str]] = {
        "Programming": [
            "python",
            "java",
            "c",
            "c++",
            "c#",
            "go",
            "rust",
            "scala",
            "kotlin",
            "javascript",
            "typescript",
            "php",
            "ruby",
            "swift",
            "dart",
            "matlab",
            "r",
            "perl",
            "lua",
            "bash",
            "shell scripting",
            "powershell",
            "sql",
            "pl/sql",
            "nosql",
            "graph queries",
            "bash scripting",
            "pandas",
            "numpy",
            "scipy",
            "sympy",
            "pytorch",
            "tensorflow",
            "keras",
            "jax",
            "flax",
            "autograd",
            "opencv",
            "scikit-learn",
            "xgboost",
            "lightgbm",
            "catboost",
            "statsmodels",
            "matplotlib",
            "seaborn",
            "plotly",
            "bokeh",
            "ggplot2",
            "tidyverse",
            "dplyr",
            "spark",
            "pyspark",
            "hadoop",
            "mapreduce",
            "apache beam",
            "stream processing",
            "apache flink",
            "apache kafka streams",
            "data structures",
            "algorithms",
            "object oriented programming",
            "functional programming",
            "multithreading",
            "concurrency",
            "asyncio",
            "multiprocessing",
            "threading",
            "multiprocessing",
            "unit testing",
            "integration testing",
            "test-driven development",
            "tdd",
            "mocking",
            "pytest",
            "unittest",
            "jest",
            "mocha",
            "nunit",
            "xunit",
            "junit",
            "test automation",
            "continuous testing",
            "coding standards",
            "code review",
            "design patterns",
            "solid principles",
            "microservices",
            "monorepo",
            "monolith",
        ],
        "Web Development": [
            "rest",
            "rest api",
            "graphql",
            "soap",
            "api design",
            "http",
            "https",
            "authentication",
            "authorization",
            "oauth",
            "openid connect",
            "jwt",
            "session management",
            "cookies",
            "caching",
            "rate limiting",
            "pagination",
            "websocket",
            "webhooks",
            "api gateway",
            "microservice api",
            "swagger",
            "openapi",
            "postman",
            "swagger ui",
            "api documentation",
            "performance optimization",
            "load testing",
            "stress testing",
            "k6",
            "jmeter",
            "locust",
            "httpie",
            "browser debugging",
            "dom",
            "html",
            "css",
            "sass",
            "less",
            "responsive design",
            "accessibility",
            "web security",
        ],
        "Frontend": [
            "react",
            "next.js",
            "vue",
            "angular",
            "svelte",
            "redux",
            "mobx",
            "zustand",
            "state management",
            "frontend",
            "tailwind css",
            "bootstrap",
            "material ui",
            "chakra ui",
            "css modules",
            "responsive ui",
            "typescript",
            "javascript",
            "webpack",
            "vite",
            "babel",
            "eslint",
            "prettier",
            "jest",
            "rtl",
            "unit testing",
        ],
        "Backend": [
            "django",
            "flask",
            "fastapi",
            "starlette",
            "spring",
            "spring boot",
            "hibernate",
            "node.js",
            "express",
            "nestjs",
            "ruby on rails",
            "dotnet",
            "asp.net",
            "asp.net core",
            "grpc",
            "thrift",
            "serialization",
            "data validation",
            "pydantic",
            "dependency injection",
            "service layer",
            "repository pattern",
            "domain driven design",
            "ddd",
            "cqrs",
            "event driven architecture",
            "message queues",
        ],
        "AI": [
            "artificial intelligence",
            "ai",
            "knowledge graphs",
            "reasoning",
            "agentic workflows",
            "agent",
            "expert systems",
            "recommendation systems",
            "information retrieval",
            "semantic search",
            "ranking",
            "recommenders",
            "computer vision",
            "image classification",
            "object detection",
            "ocr",
            "natural language processing",
            "nlp",
            "sentiment analysis",
            "text classification",
            "named entity recognition",
            "ner",
            "question answering",
            "summarization",
            "language modeling",
            "transformers",
            "attention mechanism",
            "embeddings",
            "sentence embeddings",
            "tf-idf",
            "cosine similarity",
            "similarity search",
        ],
        "ML": [
            "machine learning",
            "supervised learning",
            "unsupervised learning",
            "semi-supervised learning",
            "reinforcement learning",
            "feature engineering",
            "model selection",
            "hyperparameter tuning",
            "cross-validation",
            "grid search",
            "random search",
            "bayesian optimization",
            "regularization",
            "l1 regularization",
            "l2 regularization",
            "elastic net",
            "roc-auc",
            "precision",
            "recall",
            "f1-score",
            "confusion matrix",
            "precision-recall",
            "loss functions",
            "gradient descent",
            "stochastic gradient descent",
            "adam optimizer",
            "learning rate",
            "early stopping",
            "class imbalance",
            "smote",
            "ensemble learning",
            "bagging",
            "boosting",
            "stacking",
            "random forests",
            "decision trees",
            "logistic regression",
            "linear regression",
            "ridge regression",
            "lasso regression",
            "svm",
            "support vector machines",
            "k-nearest neighbors",
            "knn",
            "naive bayes",
            "neural networks",
            "deep learning",
            "transfer learning",
            "fine-tuning",
            "pretrained models",
            "dimensionality reduction",
            "pca",
            "svd",
            "t-sne",
            "umap",
            "topic modeling",
            "lda",
            "nmf",
            "clustering",
            "k-means",
            "hierarchical clustering",
            "dbscan",
        ],
        "Data Science": [
            "data analysis",
            "exploratory data analysis",
            "eda",
            "data cleaning",
            "data preprocessing",
            "data visualization",
            "statistics",
            "hypothesis testing",
            "confidence intervals",
            "a/b testing",
            "causal inference",
            "time series",
            "forecasting",
            "seasonality",
            "anomaly detection",
            "feature selection",
            "correlation analysis",
            "multicollinearity",
            "regression",
            "classification",
            "model evaluation",
            "data pipelines",
            "etl",
            "data warehousing",
            "data modeling",
            "star schema",
            "snowflake schema",
        ],
        "Databases": [
            "postgresql",
            "postgres",
            "mysql",
            "mariadb",
            "sqlite",
            "oracle",
            "sql server",
            "mongodb",
            "cassandra",
            "redis",
            "dynamodb",
            "neo4j",
            "elastic search",
            "elasticsearch",
            "opensearch",
            "document databases",
            "columnar databases",
            "data lake",
            "data warehouse",
            "etl jobs",
            "indexing",
            "query optimization",
            "stored procedures",
            "triggers",
            "views",
            "transactions",
            "partitioning",
            "sharding",
            "replication",
        ],
        "Cloud": [
            "aws",
            "amazon web services",
            "s3",
            "ec2",
            "lambda",
            "dynamodb",
            "rds",
            "sqs",
            "sns",
            "cloudwatch",
            "iam",
            "elastic beanstalk",
            "step functions",
            "ecr",
            "ecs",
            "eks",
            "vpc",
            "redshift",
            "cloudfront",
            "azure",
            "microsoft azure",
            "blob storage",
            "azure functions",
            "cosmos db",
            "gcp",
            "google cloud",
            "cloud storage",
            "bigquery",
            "cloud run",
            "pub/sub",
            "compute engine",
            "kubernetes",
        ],
        "DevOps": [
            "docker",
            "containerization",
            "kubernetes",
            "helm",
            "ingress",
            "ci/cd",
            "jenkins",
            "github actions",
            "gitlab ci",
            "travis ci",
            "bitbucket pipelines",
            "terraform",
            "infrastructure as code",
            "ansible",
            "puppet",
            "chef",
            "linux",
            "bash",
            "shell",
            "systemd",
            "monitoring",
            "prometheus",
            "grafana",
            "logging",
            "elk",
            "elasticsearch logstash kibana",
            "loki",
            "tempo",
            "tracing",
            "distributed tracing",
            "alerting",
            "incident management",
            "load balancer",
            "nginx",
            "apache",
            "caddy",
        ],
        "DevOps & Tooling": [
            "makefile",
            "cmake",
            "gradle",
            "maven",
            "ant",
            "sbt",
            "pip",
            "poetry",
            "conda",
            "virtualenv",
            "docker-compose",
            "kustomize",
            "artifact repository",
            "nexus",
            "jfrog artifactory",
            "semver",
            "release management",
        ],
        "Cyber Security": [
            "security",
            "threat modeling",
            "owasp",
            "owasp top 10",
            "vulnerability scanning",
            "pentesting",
            "penetration testing",
            "sdlc security",
            "secure coding",
            "input validation",
            "authentication",
            "authorization",
            "access control",
            "encryption",
            "tls",
            "ssl",
            "hashing",
            "hmac",
            "jwt security",
            "owasp api security",
            "web application security",
            "network security",
            "firewalls",
            "siem",
            "log analysis",
            "incident response",
        ],
        "Testing": [
            "unit tests",
            "integration tests",
            "e2e testing",
            "end-to-end testing",
            "selenium",
            "playwright",
            "cypress",
            "postman tests",
            "contract testing",
            "consumer driven contract",
            "mock servers",
        ],
        "Frameworks": [
            "pandas",
            "numpy",
            "scikit-learn",
            "nltk",
            "spacy",
            "fasttext",
            "gensim",
            "huggingface",
            "transformers",
            "flask",
            "django",
            "fastapi",
            "spring",
            "spring boot",
            "node.js",
            "express",
            "nestjs",
            "react",
            "next.js",
            "vue",
            "angular",
            "electron",
        ],
    }

    # Ensure at least 300+ by adding a large set of additional skills/keywords.
    # These are intentionally varied and common in resumes.
    additional = [
        # Data engineering
        "apache spark",
        "spark sql",
        "databricks",
        "delta lake",
        "aws glue",
        "azure data factory",
        "dbt",
        "airflow",
        "prefect",
        "great expectations",
        "data quality",
        "schema evolution",
        "cdc",
        "change data capture",
        "streaming etl",
        # Containers / cloud native
        "docker swarm",
        "container registry",
        "oauth2",
        # Networking
        "tcp",
        "udp",
        "http methods",
        "restful endpoints",
        "dns",
        "load balancing",
        "latency optimization",
        "bandwidth",
        # Misc tools
        "jira",
        "confluence",
        "trello",
        "git",
        "github",
        "gitlab",
        "bitbucket",
        "svn",
        "code coverage",
        "static analysis",
        "sonarqube",
        "codeql",
        "snyk",
        "owasp dependency-check",
        # NLP
        "tokenization",
        "lemmatization",
        "stemming",
        "bag of words",
        "word embeddings",
        "tf-idf vectorization",
        "word2vec",
        "glove",
        "bert",
        "roberta",
        "t5",
        "bart",
        "gpt",
        "semantic similarity",
        # ML Ops
        "mlflow",
        "model registry",
        "feature store",
        "data versioning",
        "dvc",
        "kfp",
        "kubeflow",
        "pipelines",
        "model monitoring",
        "drift detection",
        "online inference",
        "batch inference",
        # Algorithms
        "dynamic programming",
        "greedy algorithms",
        "backtracking",
        "graph algorithms",
        "bfs",
        "dfs",
        "dijkstra",
        "a*",
        "topological sort",
        # Security further
        "sast",
        "dast",
        "supply chain security",
        # DevOps more
        "linux permissions",
        "cron",
        "shell scripting",
        "system logging",
    ]

    # Add many generated variants to guarantee count.
    # These are still real/usable resume keywords.
    generated_variants = []
    base_frameworks = [
        "agile",
        "scrum",
        "kanban",
        "sdLC",
        "sdclc",
        "uml",
        "bpmn",
        "mermaid",
        "swagger",
        "openapi",
    ]
    for item in base_frameworks:
        generated_variants.extend([item, item.replace("-", " ")])

    # Merge additional into categories.
    for k in ["Data Science", "ML", "Frameworks", "DevOps", "Cloud", "Programming", "AI"]:
        if additional:
            # Distribute roughly.
            for s in additional:
                if len(db.get(k, [])) < 60:
                    db.setdefault(k, []).append(s)
                    break

    # Add generated variants to Programming.
    db.setdefault("Programming", []).extend(generated_variants)

    # De-duplicate while preserving order.
    for cat, items in db.items():
        seen: Set[str] = set()
        deduped: List[str] = []
        for it in items:
            n = _normalize_phrase(it)
            if n and n not in seen:
                seen.add(n)
                deduped.append(n)
        db[cat] = deduped

    # Validate count requirement.
    total = sum(len(v) for v in db.values())
    if total < 300:
        # Hard fail to guarantee requirement.
        raise RuntimeError(f"Skill database must contain >=300 skills, but got {total}.")

    return db


def extract_skills(text: str, *, skill_db: Dict[str, List[str]]) -> Set[str]:
    """Extract skills/keywords from the given text using phrase matching."""

    normalized = (text or "").lower()
    normalized = re.sub(r"\s+", " ", normalized)

    found: Set[str] = set()

    for category, skills in skill_db.items():
        for skill in skills:
            # Use word-boundary-ish matching: split into tokens and ensure every token appears.
            # But also handle multi-word phrases with a regex.
            phrase = _normalize_phrase(skill)
            if not phrase:
                continue

            # Convert phrase to regex with flexible whitespace.
            tokens = phrase.split()
            if len(tokens) == 1:
                token = re.escape(tokens[0])
                # Match token as a whole word.
                pattern = rf"\b{token}\b"
                if re.search(pattern, normalized):
                    found.add(phrase)
            else:
                # For phrases, match a sequence of words.
                pattern = r"\b" + r"\s+".join(re.escape(t) for t in tokens) + r"\b"
                if re.search(pattern, normalized):
                    found.add(phrase)

    return found


def skill_match_report(resume_text: str, job_text: str, *, skill_db: Dict[str, List[str]]) -> SkillMatch:
    """Compute matching/missing/extra skills."""

    resume_skills = extract_skills(resume_text, skill_db=skill_db)
    job_skills = extract_skills(job_text, skill_db=skill_db)

    matching = sorted(job_skills.intersection(resume_skills))
    missing = sorted(job_skills.difference(resume_skills))
    extra = sorted(resume_skills.difference(job_skills))

    by_category: Dict[str, List[str]] = {}
    # Build reverse mapping to category.
    skill_to_cat: Dict[str, str] = {}
    for cat, skills in skill_db.items():
        for s in skills:
            skill_to_cat[s] = cat

    for s in matching:
        cat = skill_to_cat.get(s, "Other")
        by_category.setdefault(cat, []).append(s)

    return SkillMatch(
        matching_skills=matching,
        missing_skills=missing,
        extra_skills=extra,
        by_category={k: sorted(v) for k, v in by_category.items()},
    )

