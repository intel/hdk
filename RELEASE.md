# HDK Release Process

HDK uses a semantic versioning scheme for software version numbers. The version number consists of MAJOR.MINOR.PATCH. The Major version increments each time the API changes. Minor versions indicate a new branch from the main code branch. Patch versions increment when there is a release from an existing release branch. Typically, patch versions will be released for bug fixes, where minor versions will include more substantive changes. 

### Release Checklist

1. Create a new release branch with the name `vMAJOR.MINOR.PATCH`.
    * For major and minor version releases, create a branch from `main`.
    * For patch version releases, create a branch from the existing release branch on which the patch is going to be based (e.g. `v0.1.1` would be branched from `v0.1.0`).
2. Bump the version number in `CMakeLists.txt` in the release branch.
3. Ensure CI on the release branch is green.
4. Create a tag for the release. 
5. Create GitHub Release w/ source code archive. 
6. Build and publish updated `conda-forge` packages.
