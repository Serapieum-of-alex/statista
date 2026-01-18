# Changelog


## 0.7.0 (2026-01-18)


- build: migrate from poetry to uv package manager (#128)
- build: migrate from poetry to uv package manager
-   - Replace Poetry with uv as the primary package manager
  - Convert project.optional-dependencies to dependency-groups (PEP 735)
  - Update all CI/CD workflows to use uv with composite actions
  - Consolidate release workflows (remove release-bump.yml and github-release.yml)
  - Update actions to latest versions (checkout@v5, composite actions v1)
  - Add comprehensive mypy configuration with module-specific overrides
  - Fix deprecated pandas offset alias "A-OCT" to "YE-OCT" in eva.py
  - Simplify mkdocs deployment workflow with reusable composite actions
  - Update PyPI release workflow to use uv build
- ci(release-bump): update step name to clarify version bump and tagging process
- ci(release-bump): remove GitHub token from release-bump workflow
- ci(release-bump): add GitHub token for release bump in main
- chore(pyproject): remove update_changelog_on_bump configuration
- chore(release): add Commitizen and GitHub release workflow (#126)
- chore(release): add Commitizen and GitHub release workflow
- - Added Commitizen for conventional commit management and versioning
- Created GitHub Actions workflow to publish releases using Commitizen
- Updated Commitizen to v4.8.3 and replaced deprecated commands
- Removed `dependabot.yml` and Poetry configuration
- ref: #127
- chore(templates): refine issue templates and add performance and documentation categories (#124)
- - Updated bug report template with clearer sections and reproducibility requirements
- Enhanced feature request template with detailed prompts for API changes and performance considerations
- Added new templates for performance issues and documentation improvements
- Configured default behavior to disable blank issues and added links for Q&A and documentation access
- ref: #125
- ci: update workflow triggers and improve Codecov reporting (#122)
- ci: update workflow triggers and improve Codecov reporting
- - Trigger PyPI publish workflow only on release published event
- Adjust tests.yml trigger to fix Codecov reporting for main branch
- Added step to upload detailed Codecov test report
- Removed Codecov token and restricted trigger to pushes on main
- ref: #123
- ci(pypi): trigger publish workflow only on release published event (#120)
- ref: #121

## 0.6.3 (2025-08-08)
##### Distributions
* fix the `chisquare` method to all distributions.

## 0.6.2 (2025-07-31)
##### Docs
* add complete documentation for all modules.

#### Dev
* refactor all modules.
* fix pre-commit hooks.


## 0.6.1 (2025-06-03)
##### Dev
* replace the setup.py with pyproject.toml.
* migrate the documentation to use mkdocs-material.
* add complete documentation for all modules.


## 0.6.0 (2024-08-18)

##### dev
* Add documentations for the `distributions`, and `eva` modules.
* Add autodoc for all modules.
* Test docstrings as part of CI and pre-commit hooks.
* Test notebooks as part of CI.
* Simplify test for the distributions module

##### distributions
* move the `cdf` and `parameters` for all the methods to be optional parameters.
* rename `theoretical_estimate` method to `inverse_cdf`.
* All distributions can be instantiated with the parameters and/or data.
* rename the `probability_plot` method to `plot`.
* move the `confidence_interval` plot from the `probability_plot/plot` to the method `confidence_interval` and can be
  called by activating the `plot_figure=True`.

##### descriptors
* rename the `metrics` module to `descriptors`.

## 0.5.0 (2023-12-11)

* Unify the all the methods for the distributions.
* Use factory design pattern to create the distributions.
* add tests for the eva module.
* use snake_case for the methods and variables.

## 0.4.0 (2023-11-23)

* add Pearson 3 distribution
* Use setup.py instead of pyproject.toml.
* Correct pearson correlation coefficient and add documentation .
* replace the pdf and cdf by the methods from scipy package.

## 0.3.0 (2023-02-19)

* add documentations for both GEV and gumbel distributions.
* add lmoment parameter estimation method for all distributions.
* add exponential and normal distributions
* modify the pdf, cdf, and probability plot plots
* create separate plot and confidence_interval modules.

## 0.2.0 (2023-02-08)

* add eva (Extreme value analysis) module
* fix bug in obtaining distribution parameters using optimization method

## 0.1.8 (2023-01-31)

* bump up versions

## 0.1.7 (2022-12-26)

* lock numpy to version 1.23.5

## 0.1.0 (2022-05-24)

* First release on PyPI.
