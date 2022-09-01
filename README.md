# Performance Scorecard Generator
A tool to report out performance metrics for classification and regression models to technical and non-technical audiences.

The U.S. Census Bureau has a mission “to serve as the nation’s leading provider of quality data about its people and economy.” To meet this mission, the bureau aims to perform highly accurate surveys. One method to increase this accuracy is to maintain a Master Address File (MAF) to ensure each residence is counted in the census. The Geography Division is implementing emerging technology to leverage satellite imagery to update the MAF through the Automated Change Detection Core (ACDC); utilizing machine learning to detect change in residences. This effort has implications to variations in counting and representation of people in the census and requires analysis of the performance to evaluate the model for bias: the systemic, statistical and computational, and human bias that is injected into the model at all steps in the design, testing, and deployment process. How do we systematically assess the performance of our models in order to investigate bias? How do we pave the way for new standards of practice in the Census Bureau and beyond?

To help address the problem outlined above, a tool was built to address several goals of the federal government's mission for AI and machine learning models: performance driven, transparent, accountable, understandable, and accurate. When used in the machine learning workflow of the ACDC, this tool ensures that decisions are made based on the performance of the model by reporting in a format that is understandable and readable to technical and non-technical stakeholders. This tool creates accountability for model authors to report on their efforts and to meet certain standards of performance as decided by their team. By implementing this tool, the ACDC and Geography Division become leaders in new documentation methods and pave the way for a more transparent and accountable U.S. Census Bureau in AI and machine learning.


## Tool Description 
A performance scorecard is a piece of documentation built to report a model’s metrics of performance for the entire model and across categories of interest. This is an emerging documentation tool in machine learning, currently in development by Microsoft Azure as a component of their Responsible AI dashboard. After a regression or classification model has been trained, the model author should generate a performance scorecard to evaluate the metrics of performance for the model and across categories of interest. The information provided should be evaluated to guide conversation on the performance of a model to investigate sources of bias. The scorecard should be re-run with new iterations of the model and shared with stakeholders in decisions regarding a model’s deployment.

## Installation

This package requires Python >= 3.6.

Clone the repo and install the package:
```
git clone git@github.com:me25hage/performance-card-generator.git
```

Download dependencies.
```
pip install numpy
```
```
pip install scikit-image
pip install scikit-learn
```

## Features

* [`performanceCard.py`](performance-card-generator/performanceCard.py): Main module that creates performance scorecard.

## Usage

This command line tool creates a python shell in which the user is prompted to answer a series of questions regarding their model.

The script will generate a model card from the user's responses input through the command line or through a .txt document. To input answers through a .txt document. Begin the tool and answer the responses as prompted, selecting 'Text Document' as your input preference.
```
Usage: python ModelCardGenerator.py

  Open shell to generate a model card.

```

## Questions & Issues

To suggest changes to the available metrics, please [open an issue](https://github.com/me25hage/performance-card-generator/issues).
