from .algorithms import (
    M1Algorithm,
    M2Algorithm,
    generateCARs,
    createCARs,
    top_rules
)
from .data_structures import TransactionDB


class CBA():
    """Class for training a testing the
    CBA Algorithm.

    Parameters:
    -----------
    support : float
    confidence : float
    algorithm : string
        Algorithm for building a classifier.
    maxlen : int
        maximum length of mined rules
    """

    def __init__(self, support=0.10, confidence=0.5, maxlen=10, algorithm="m1"):
        if algorithm not in ["m1", "m2"]:
            raise Exception("algorithm parameter must be either 'm1' or 'm2'")
        if 0 > support or support > 1:
            raise Exception("support must be on the interval <0;1>")
        if 0 > confidence or confidence > 1:
            raise Exception("confidence must be on the interval <0;1>")
        if maxlen < 1:
            raise Exception("maxlen cannot be negative or 0")

        self.support = support * 100
        self.confidence = confidence * 100
        self.algorithm = algorithm
        self.maxlen = maxlen
        self.clf = None
        self.target_class = None

        self.available_algorithms = {
            "m1": M1Algorithm,
            "m2": M2Algorithm
        }

    def rule_model_accuracy(self, txns):
        """Takes a TransactionDB and outputs
        accuracy of the classifier
        """
        if not self.clf:
            raise Exception("CBA must be trained using fit method first")
        if not isinstance(txns, TransactionDB):
            raise Exception("txns must be of type TransactionDB")

        return self.clf.test_transactions(txns)

    def rule_model_confusion_matrix(self, actual, predicted):
        """
        Takes predicted and actual amd aoutputs confusion matrix of the classifier.
        The predicted and actual calculates as follows:
            predicted = self.predict_all(txns)
            actual = txns.classes

        :param actual:
        :param predicted:
        :return:
        """
        return self.clf.confusion_matrix(actual, predicted)

    def rule_model_classification_report(self, actual, predicted):
        """
        Takes predicted and actual amd outputs classification reports of the classifier.
        The predicted and actual calculates as follows:
            predicted = self.predict_all(txns)
            actual = txns.classes

        :param actual:
        :param predicted:
        :return:
        """
        return self.clf.classification_report(actual, predicted)

    def fit(self, transactions, top_rules=[], top_rules_args={}):
        """Trains the model based on input transaction
        and returns self.
        """
        if not isinstance(transactions, TransactionDB):
            raise Exception("transactions must be of type TransactionDB")
        self.target_class = transactions.header[-1]
        used_algorithm = self.available_algorithms[self.algorithm]
        cars = None
        if len(top_rules) > 0:
            cars = createCARs(top_rules)
            print(len(top_rules))
        elif not top_rules_args:
            cars = generateCARs(transactions, support=self.support, confidence=self.confidence, maxlen=self.maxlen)
        else:
            rules = top_rules(transactions.string_representation, appearance=transactions.appeardict, **top_rules_args)
            cars = createCARs(rules)
        self.clf = used_algorithm(cars, transactions).build()
        return self

    def predict(self, X):
        """Method that can be used for predicting
        classes of unseen cases.

        CBA.fit must be used before predicting.
        """
        if not self.clf:
            raise Exception("CBA must be train using fit method first")

        if not isinstance(X, TransactionDB):
            raise Exception("X must be of type TransactionDB")

        return self.clf.predict_all(X)

    def predict_probability(self, X):
        """Method for predicting probablity of
        given classification
¨
        CBA.fit must be used before predicting probablity.
        """

        return self.clf.predict_probability_all(X)

    def predict_matched_rules(self, X):
        """for each data instance, returns a rule that
        matched it according to the CBA order (sorted by
        confidence, support and length)
        """

        return self.clf.predict_matched_rule_all(X)

    def update_cba_model(self, n1, new_transactions, use_top_rules=False):
        """
        This function updates CBA model with new coming training data. If use_top_rules parameter is True, top_rules
        function uses to generate rules.

        Parameters
        ----------
        n1 : integer
        new_transactions : TransactionDB
        use_top_rules : boolean
        :return:
        """
        print(self.support, self.confidence)
        cba = CBA(support=float(self.support / 100), confidence=float(self.confidence / 100), algorithm="m1")

        # number of new coming rules
        n2 = len(new_transactions)
        if (use_top_rules):
            tr = top_rules(new_transactions.string_representation)
            cba.fit(new_transactions, top_rules=tr)
        else:
            cba.fit(new_transactions)
        # update rule metric if rules are same, otherwise add new rule.
        new_rules = list()
        for i in range(len(cba.clf.rules)):
            is_new_lhs_rhs = True
            for j in range(0, len(self.clf.rules)):
                sup1 = self.clf.rules[j].support
                sup2 = cba.clf.rules[i].support
                conf1 = self.clf.rules[j].confidence
                conf2 = cba.clf.rules[i].confidence
                if self.clf.rules[j].antecedent.string() == cba.clf.rules[i].antecedent.string() and self.clf.rules[
                    j].consequent.string() == cba.clf.rules[i].consequent.string():
                    is_new_lhs_rhs = False
                    self.clf.rules[j].support = self.update_support(sup1, sup2, n1, n2)
                    self.clf.rules[j].confidence = self.update_confidence(sup1, sup2, conf1, conf2, n1, n2)
                    print("Updated rule: {} with conf:{} and support:{}".
                          format(self.clf.rules[j], self.clf.rules[j].support, self.clf.rules[j].confidence))
                elif self.clf.rules[j].antecedent.string() == cba.clf.rules[i].antecedent.string() and self.clf.rules[
                    j].consequent.string() != cba.clf.rules[i].consequent.string():
                    is_new_lhs_rhs = False
                    freq1 = n1 * sup1
                    freq2 = n2 * sup2
                    # keep rule has greater frequency
                    if (freq1 >= freq2):
                        self.clf.rules[j].support = self.update_support(sup1, sup2, n1, n2)
                        self.clf.rules[j].confidence = self.update_confidence(sup1, sup2, conf1, conf2, n1, n2)
                        print("Updated rule: {} with conf:{} and support:{}".
                              format(self.clf.rules[j], self.clf.rules[j].support, self.clf.rules[j].confidence))
                    else:
                        # if the new rule has the same lhs and different rhs with any older rules.
                        cba.clf.rules[i].support = self.update_new_rule_support(sup2, n1, n2)
                        cba.clf.rules[i].confidence = self.update_new_rule_confidence(sup1, sup2, conf1, conf2, n1, n2)
                        self.clf.rules[j] = cba.clf.rules[i]
            # if the lhs and rhs are new
            if is_new_lhs_rhs:
                cba.clf.rules[i].support = self.update_new_rule_support(sup2, n1, n2)
                new_rules.append(cba.clf.rules[i])
        self.clf.rules += new_rules
        new_rules.clear()
        self.clf.rules.sort(reverse=True)

    def update_cba_model2(self,  model_list):
        main_model = model_list[0]["model"]
        self.size = model_list[0]["size"]
        for cba_model in model_list[1:]:
            n2 = cba_model["size"]
            new_rules = list()
            for i in range(len(cba_model["model"].clf.rules)):
                is_new_lhs_rhs = True
                for j in range(0, len(main_model.clf.rules)):
                    sup1 = main_model.clf.rules[j].support
                    sup2 = cba_model["model"].clf.rules[i].support
                    conf1 = main_model.clf.rules[j].confidence
                    conf2 = cba_model["model"].clf.rules[i].confidence
                    if main_model.clf.rules[j].antecedent.string() == cba_model["model"].clf.rules[i].antecedent.string() and main_model.clf.rules[
                        j].consequent.string() == cba_model["model"].clf.rules[i].consequent.string():
                        is_new_lhs_rhs = False
                        main_model.clf.rules[j].support = self.update_support(sup1, sup2, self.size, n2)
                        main_model.clf.rules[j].confidence = self.update_confidence(sup1, sup2, conf1, conf2, self.size, n2)
                        print("Updated rule: {} with conf:{} and support:{}".
                            format(main_model.clf.rules[j], main_model.clf.rules[j].support, main_model.clf.rules[j].confidence))
                    elif main_model.clf.rules[j].antecedent.string() == cba_model["model"].clf.rules[i].antecedent.string() and main_model.clf.rules[
                        j].consequent.string() != cba_model["model"].clf.rules[i].consequent.string():
                        is_new_lhs_rhs = False
                        freq1 = self.size * sup1
                        freq2 = n2 * sup2
                        # keep rule has greater frequency
                        if (freq1 >= freq2):
                            main_model.clf.rules[j].support = self.update_support(sup1, sup2, self.size, n2)
                            main_model.clf.rules[j].confidence = self.update_confidence(sup1, sup2, conf1, conf2, self.size, n2)
                            print("Updated rule: {} with conf:{} and support:{}".
                                format(main_model.clf.rules[j], main_model.clf.rules[j].support, main_model.clf.rules[j].confidence))
                        else:
                            # if the new rule has the same lhs and different rhs with any older rules.
                            cba_model["model"].clf.rules[i].support = self.update_new_rule_support(sup2, self.size, n2)
                            cba_model["model"].clf.rules[i].confidence = self.update_new_rule_confidence(sup1, sup2, conf1, conf2, self.size, n2)
                            main_model.clf.rules[j] = cba_model["model"].clf.rules[i]
                # if the lhs and rhs are new
                if is_new_lhs_rhs:
                    cba_model["model"].clf.rules[i].support = self.update_new_rule_support(sup2, self.size, n2)
                    new_rules.append(cba_model["model"].clf.rules[i])
            main_model.clf.rules += new_rules
            self.size = self.size + n2
            new_rules.clear()
            main_model.clf.rules.sort(reverse=True)
            self.clf = main_model.clf
            return main_model

    def update_support(self, sup1, sup2, n1, n2):
        """
        This function updates support metric of the rule.
        Parameters:
        ----------------------
            :param sup1: float - support of older rules
            :param sup2: float - support of new coming rules
            :param n1: integer - number of older rules
            :param n2: integer - number of new coming rules
            :return: new_support: float
        """

        new_support = (sup1 * n1 + sup2 * n2) / (n1 + n2)

        return new_support

    def update_confidence(self, sup1, sup2, conf1, conf2, n1, n2):
        """
        This function updates confidence metric of the rule.
        Parameters:
        ----------------------
            :param sup1: float - support of older rule
            :param sup2: float - support of new coming rule
            :param conf1: float - confidence of older rule
            :param conf2: float - confidence of new coming rule
            :param n1: integer - number of older rules
            :param n2: integer - number of new coming rules
            :return: new_confidence: float
        """
        new_confidence = ((sup1 * n1 + sup2 * n2) * conf1 * conf2) / (sup1 * n1 * conf2 + sup2 * n2 * conf1)
        return new_confidence

    def update_new_rule_support(self, sup1, n1, n2):
        """
        This function updates support metric of the appended rule.
        Parameters:
        ----------------------
            :param sup1: float - support of older rule
            :param n1: integer - number of older rules
            :param n2: integer - number of new coming rules
            :return: new_support: float
        """
        new_support = sup1 * n1 / (n1 + n2)
        return new_support

    def update_new_rule_confidence(self, sup1, sup2, conf1, conf2, n1, n2):
        """
        This function updates confidence metric of the appended rule.
                Parameters:
                ----------------------
                    :param sup1: float - support of older rule
                    :param sup2: float - support of new coming rule
                    :param conf1: float - confidence of older rule
                    :param conf2: float - confidence of new coming rule
                    :param n1: integer - number of older rules
                    :param n2: integer - number of new coming rules
                    :return: new_confidence: float
                """
        new_confidence = (sup2 * n2 * conf1 * conf2) / (sup1 * n1 * conf2 + sup2 * n2 * conf1)
        return new_confidence