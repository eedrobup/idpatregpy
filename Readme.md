<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![BSD 3-Clause License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!--
  <a href="https://github.com/eedrobup/idpatregpy">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
  -->

<h3 align="center">IDentity PATtern REcoGnition in Python</h3>

  <p align="center">
    A program for identity classification by patterns in images using landmark-points-extracted patch and image processing.<br />
    <br />
    <a href="https://github.com/eedrobup/idpatregpy"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/eedrobup/idpatregpy">View Demo</a>
    ·
    <a href="https://github.com/eedrobup/idpatregpy/issues">Report Bug</a>
    ·
    <a href="https://github.com/eedrobup/idpatregpy/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#supervisors">Supervisors</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This program is refactored from my master thesis at the University of Manchester in 2023 titled
#### "Identifying individual Xenopus using image analysis and machine learning."

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python][Python.org]][Python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

use `pip install -r requirements.txt` if Normal installation below does not work

### Installation

`pip install git+https://github.com/eedrobup/idpatregpy.git`

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Imagepoints object for a set of image and its landmarks
    - [x] base object
    - [ ] pre-labelled set object
    - [ ] non-labelled set object
- [ ] Bulk object for group up imagepoints object for landmarkdetector model training and identities database implementation
    - [x] base object
    - [ ] U-Net3 training object
    - [ ] Database object
- [ ] Landmarkdetector model
    - [ ] base object

See the [open issues](https://github.com/eedrobup/idpatregpy/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the Modified BSD License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Pubordee Aussavavirojekul (eedrobup) - [@pu_aus](https://twitter.com/pu_aus) - pubordee.a@gmail.com
[![LinkedIn][linkedin-shield]][linkedin-url]

Project Link: [https://github.com/eedrobup/idpatregpy](https://github.com/eedrobup/idpatregpy)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- SUPERVISORS -->
## Supervisors

* [Professor Tim Cootes](https://personalpages.manchester.ac.uk/staff/timothy.f.cootes/)
* [Dr Sarah Woolner ](https://research.manchester.ac.uk/en/persons/sarah.woolner)

<!-- ACKNOWLEDGEMENT -->
## Acknowledgments

* [Dr Nawseen Tarannum](https://www.linkedin.com/in/nawseen-tarannum-5411a76b)
* [Matthew Coates](https://www.linkedin.com/in/matthew-coates-5768715b)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/eedrobup/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/eedrobup/idpatregpy/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/eedrobup/idpatregpy.svg?style=for-the-badge
[forks-url]: https://github.com/eedrobup/idpatregpy/network/members
[stars-shield]: https://img.shields.io/github/stars/eedrobup/idpatregpy.svg?style=for-the-badge
[stars-url]: https://github.com/eedrobup/idpatregpy/stargazers
[issues-shield]: https://img.shields.io/github/issues/eedrobup/idpatregpy.svg?style=for-the-badge
[issues-url]: https://github.com/eedrobup/idpatregpy/issues
[license-shield]: https://img.shields.io/badge/License-BSD_3--Clause-orange.svg?style=for-the-badge
[license-url]: https://opensource.org/licenses/BSD-3-Clause
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/pubordee-aussavavirojekul-5bb0b611a
[Python.org]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/