from unittest import TestCase

import tiktoken

from openaivec.util import TextChunker


class TestTextChunker(TestCase):
    def setUp(self):
        self.sep = TextChunker(
            enc=tiktoken.encoding_for_model("text-embedding-3-large"),
        )

    def test_split(self):
        text = """
Kubernetes was announced by Google on June 6, 2014.[10] The project was conceived and created by Google employees Joe Beda, Brendan Burns, and Craig McLuckie. Others at Google soon joined to help build the project including Ville Aikas, Dawn Chen, Brian Grant, Tim Hockin, and Daniel Smith.[11][12] Other companies such as Red Hat and CoreOS joined the effort soon after, with notable contributors such as Clayton Coleman and Kelsey Hightower.[10]

The design and development of Kubernetes was inspired by Google's Borg cluster manager and based on Promise Theory.[13][14] Many of its top contributors had previously worked on Borg;[15][16] they codenamed Kubernetes "Project 7" after the Star Trek ex-Borg character Seven of Nine[17] and gave its logo a seven-spoked ship's wheel (designed by Tim Hockin). Unlike Borg, which was written in C++,[15] Kubernetes is written in the Go language.

Kubernetes was announced in June, 2014 and version 1.0 was released on July 21, 2015.[18] Google worked with the Linux Foundation to form the Cloud Native Computing Foundation (CNCF)[19] and offered Kubernetes as the seed technology.

Google was already offering a managed Kubernetes service, GKE, and Red Hat was supporting Kubernetes as part of OpenShift since the inception of the Kubernetes project in 2014.[20] In 2017, the principal competitors rallied around Kubernetes and announced adding native support for it:

VMware (proponent of Pivotal Cloud Foundry)[21] in August,
Mesosphere, Inc. (proponent of Marathon and Mesos)[22] in September,
Docker, Inc. (proponent of Docker)[23] in October,
Microsoft Azure[24] also in October,
AWS announced support for Kubernetes via the Elastic Kubernetes Service (EKS)[25] in November.
Cisco Elastic Kubernetes Service (EKS)[26] in November.
On March 6, 2018, Kubernetes Project reached ninth place in the list of GitHub projects by the number of commits, and second place in authors and issues, after the Linux kernel.[27]

Until version 1.18, Kubernetes followed an N-2 support policy, meaning that the three most recent minor versions receive security updates and bug fixes.[28] Starting with version 1.19, Kubernetes follows an N-3 support policy.[29]
"""

        chunks = self.sep.split(text, max_tokens=256, sep=[".", "\n\n"])

        # Assert that the number of chunks is as expected
        enc = tiktoken.encoding_for_model("text-embedding-3-large")

        for chunk in chunks:
            self.assertLessEqual(len(enc.encode(chunk)), 256)
