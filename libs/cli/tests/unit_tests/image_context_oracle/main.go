// Command image_context_oracle applies the Dockerfile ADD/COPY instructions
// the LangGraph CLI emits to a local build context, using the same inclusion
// rules as Docker's default Linux builder.
//
// Buildx (the default `docker build` since Docker CLI v29, cmd/docker/builder.go)
// sends the context directory through fsutil.NewFS, which walks every file,
// then BuildKit drops paths matched by the dockerignore patterns. Those
// patterns come from <Dockerfile>.dockerignore when the Dockerfile is a real
// file, otherwise from <context>/.dockerignore. A missing file means an empty
// exclude list, so dotfiles such as .env are included. ADD/COPY of a directory
// then copies the remaining contents into the image. The legacy builder
// (DOCKER_BUILDKIT=0) packs the same directory with TarWithOptions and the
// same exclude list.
//
// Named build contexts use only that directory's .dockerignore.
// COPY/ADD --exclude patterns are applied on top, to the paths being copied.
// RUN instructions are not executed: a file that was copied stays recorded,
// because deleting it in a later layer leaves it in the image history.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/moby/patternmatcher"
	"github.com/moby/patternmatcher/ignorefile"
)

type request struct {
	Dockerfile     string            `json:"dockerfile"`
	DockerfilePath string            `json:"dockerfile_path"`
	Context        string            `json:"context"`
	NamedContexts  map[string]string `json:"named_contexts"`
}

type copiedFile struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

type response struct {
	Files []copiedFile `json:"files"`
	Error string       `json:"error,omitempty"`
}

func main() {
	var req request
	if err := json.NewDecoder(os.Stdin).Decode(&req); err != nil {
		fail(err)
	}
	files, err := apply(req)
	if err != nil {
		fail(err)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(response{Files: files}); err != nil {
		fail(err)
	}
}

func fail(err error) {
	enc := json.NewEncoder(os.Stdout)
	_ = enc.Encode(response{Error: err.Error()})
	os.Exit(1)
}

func apply(req request) ([]copiedFile, error) {
	if req.Context == "" {
		return nil, fmt.Errorf("context is required")
	}
	contextDir, err := filepath.Abs(req.Context)
	if err != nil {
		return nil, err
	}
	named := map[string]string{}
	for name, dir := range req.NamedContexts {
		abs, err := filepath.Abs(dir)
		if err != nil {
			return nil, err
		}
		named[name] = abs
	}

	mainExcludes, err := loadExcludes(contextDir, req.DockerfilePath, true)
	if err != nil {
		return nil, err
	}
	mainMatcher, err := newMatcher(mainExcludes)
	if err != nil {
		return nil, err
	}

	instructions, err := parseCopies(req.Dockerfile)
	if err != nil {
		return nil, err
	}

	var copied []copiedFile
	for _, inst := range instructions {
		srcDir := contextDir
		srcMatcher := mainMatcher
		if inst.from != "" {
			dir, ok := named[inst.from]
			if !ok {
				return nil, fmt.Errorf("COPY/ADD --from=%s is not a named build context", inst.from)
			}
			srcDir = dir
			excludes, err := loadExcludes(dir, "", false)
			if err != nil {
				return nil, err
			}
			srcMatcher, err = newMatcher(excludes)
			if err != nil {
				return nil, err
			}
		}
		instMatcher, err := newMatcher(inst.excludes)
		if err != nil {
			return nil, err
		}
		files, err := copyInstruction(srcDir, srcMatcher, instMatcher, inst)
		if err != nil {
			return nil, err
		}
		copied = append(copied, files...)
	}
	if copied == nil {
		copied = []copiedFile{}
	}
	return copied, nil
}

// loadExcludes implements BuildKit's dockerignore selection.
// A Dockerfile path other than stdin prefers <dockerfile>.dockerignore and
// does not merge the context file. Otherwise only <dir>/.dockerignore is read.
// A missing file yields no patterns, which excludes nothing.
func loadExcludes(contextDir, dockerfilePath string, allowSidecar bool) ([]string, error) {
	if allowSidecar && dockerfilePath != "" && dockerfilePath != "-" {
		sidecar := dockerfilePath + ".dockerignore"
		if st, err := os.Stat(sidecar); err == nil && !st.IsDir() {
			return readIgnore(sidecar)
		}
	}
	candidate := filepath.Join(contextDir, ".dockerignore")
	st, err := os.Stat(candidate)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	if st.IsDir() {
		return nil, nil
	}
	return readIgnore(candidate)
}

func readIgnore(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	patterns, err := ignorefile.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	return patterns, nil
}

func newMatcher(patterns []string) (*patternmatcher.PatternMatcher, error) {
	if len(patterns) == 0 {
		return nil, nil
	}
	return patternmatcher.New(patterns)
}

func matcherHasExclusions(pm *patternmatcher.PatternMatcher) bool {
	return pm != nil && pm.Exclusions()
}

func excluded(pm *patternmatcher.PatternMatcher, rel string) (bool, error) {
	if pm == nil {
		return false, nil
	}
	rel = filepath.ToSlash(filepath.Clean(rel))
	if rel == "." {
		return false, nil
	}
	return pm.MatchesOrParentMatches(rel)
}

type instruction struct {
	from     string
	excludes []string
	sources  []string
	dest     string
}

func parseCopies(dockerfile string) ([]instruction, error) {
	text := strings.ReplaceAll(dockerfile, "\r\n", "\n")
	var logical []string
	var b strings.Builder
	for _, line := range strings.Split(text, "\n") {
		trim := strings.TrimRight(line, " \t")
		if strings.HasSuffix(trim, "\\") {
			b.WriteString(strings.TrimSuffix(trim, "\\"))
			b.WriteByte(' ')
			continue
		}
		b.WriteString(line)
		logical = append(logical, b.String())
		b.Reset()
	}
	if b.Len() > 0 {
		logical = append(logical, b.String())
	}

	var out []instruction
	for _, line := range logical {
		trim := strings.TrimSpace(line)
		if trim == "" || strings.HasPrefix(trim, "#") {
			continue
		}
		fields, err := splitWords(trim)
		if err != nil {
			return nil, err
		}
		if len(fields) == 0 {
			continue
		}
		op := strings.ToUpper(fields[0])
		if op != "ADD" && op != "COPY" {
			continue
		}
		inst := instruction{}
		i := 1
		for i < len(fields) && strings.HasPrefix(fields[i], "--") {
			flag := fields[i]
			i++
			name, value, hasValue := strings.Cut(strings.TrimPrefix(flag, "--"), "=")
			if !hasValue {
				if i >= len(fields) {
					return nil, fmt.Errorf("flag --%s needs a value", name)
				}
				value = fields[i]
				i++
			}
			switch name {
			case "from":
				inst.from = value
			case "exclude":
				inst.excludes = append(inst.excludes, value)
			case "chown", "chmod", "checksum", "link", "parents", "keep-git-dir", "unpack":
				// These flags do not change which files are copied.
			default:
				return nil, fmt.Errorf("unsupported %s flag --%s", op, name)
			}
		}
		if len(fields)-i < 2 {
			return nil, fmt.Errorf("%s needs a source and a destination", op)
		}
		inst.sources = fields[i : len(fields)-1]
		inst.dest = fields[len(fields)-1]
		for _, src := range inst.sources {
			if strings.ContainsAny(src, "*?[") {
				return nil, fmt.Errorf("wildcard source %q is not implemented", src)
			}
		}
		out = append(out, inst)
	}
	return out, nil
}

func splitWords(line string) ([]string, error) {
	var words []string
	var b strings.Builder
	inQuote := byte(0)
	escaped := false
	flush := func() {
		if b.Len() > 0 || inQuote != 0 {
			words = append(words, b.String())
			b.Reset()
		}
	}
	for i := 0; i < len(line); i++ {
		c := line[i]
		if escaped {
			b.WriteByte(c)
			escaped = false
			continue
		}
		if c == '\\' && inQuote != '\'' {
			escaped = true
			continue
		}
		if inQuote != 0 {
			if c == inQuote {
				inQuote = 0
				continue
			}
			b.WriteByte(c)
			continue
		}
		if c == '"' || c == '\'' {
			inQuote = c
			continue
		}
		if c == ' ' || c == '\t' {
			if b.Len() > 0 {
				words = append(words, b.String())
				b.Reset()
			}
			continue
		}
		b.WriteByte(c)
	}
	if escaped || inQuote != 0 {
		return nil, fmt.Errorf("unterminated word in Dockerfile line %q", line)
	}
	flush()
	return words, nil
}

func copyInstruction(contextDir string, contextMatcher, instMatcher *patternmatcher.PatternMatcher, inst instruction) ([]copiedFile, error) {
	var copied []copiedFile
	destIsDir := strings.HasSuffix(inst.dest, "/") || len(inst.sources) > 1
	for _, src := range inst.sources {
		relSrc := cleanRel(src)
		if relSrc == ".." || strings.HasPrefix(relSrc, "../") {
			return nil, fmt.Errorf("source %q escapes the build context", src)
		}
		host := contextDir
		if relSrc != "." {
			host = filepath.Join(contextDir, filepath.FromSlash(relSrc))
		}
		info, err := os.Lstat(host)
		if err != nil {
			return nil, fmt.Errorf("source %q: %w", src, err)
		}
		skip, err := excluded(contextMatcher, relSrc)
		if err != nil {
			return nil, err
		}
		if skip {
			return nil, fmt.Errorf("source %q is excluded by dockerignore", src)
		}
		if info.Mode()&os.ModeSymlink != 0 {
			info, err = os.Stat(host)
			if err != nil {
				return nil, err
			}
		}
		if info.IsDir() {
			err = filepath.WalkDir(host, func(hostPath string, d os.DirEntry, walkErr error) error {
				if walkErr != nil {
					return walkErr
				}
				rel, err := filepath.Rel(contextDir, hostPath)
				if err != nil {
					return err
				}
				rel = filepath.ToSlash(rel)
				if rel == "." {
					return nil
				}
				skip, err := excluded(contextMatcher, rel)
				if err != nil {
					return err
				}
				relWithin := rel
				if relSrc != "." {
					relWithin = strings.TrimPrefix(rel, relSrc+"/")
				}
				instSkip := false
				if !skip {
					instSkip, err = excluded(instMatcher, relWithin)
					if err != nil {
						return err
					}
				}
				if skip || instSkip {
					if d.IsDir() && !matcherHasExclusions(contextMatcher) && !matcherHasExclusions(instMatcher) {
						return filepath.SkipDir
					}
					return nil
				}
				if d.IsDir() {
					return nil
				}
				if d.Type()&os.ModeSymlink != 0 {
					target, err := os.Stat(hostPath)
					if err != nil {
						return err
					}
					if target.IsDir() {
						return nil
					}
				}
				imagePath := joinDest(inst.dest, relWithin)
				file, err := hashFile(hostPath, imagePath)
				if err != nil {
					return err
				}
				copied = append(copied, file)
				return nil
			})
			if err != nil {
				return nil, err
			}
			continue
		}
		relWithin := path.Base(relSrc)
		skip, err = excluded(instMatcher, relWithin)
		if err != nil {
			return nil, err
		}
		if skip {
			continue
		}
		imagePath := inst.dest
		if destIsDir {
			imagePath = joinDest(inst.dest, relWithin)
		}
		file, err := hashFile(host, imagePath)
		if err != nil {
			return nil, err
		}
		copied = append(copied, file)
	}
	return copied, nil
}

func cleanRel(src string) string {
	src = filepath.ToSlash(src)
	for strings.HasPrefix(src, "./") {
		src = strings.TrimPrefix(src, "./")
	}
	if src == "" {
		return "."
	}
	return path.Clean(src)
}

func joinDest(dest, rel string) string {
	dest = path.Clean(filepath.ToSlash(dest))
	if rel == "." || rel == "" {
		return dest
	}
	return path.Clean(dest + "/" + rel)
}

func hashFile(hostPath, imagePath string) (copiedFile, error) {
	f, err := os.Open(hostPath)
	if err != nil {
		return copiedFile{}, err
	}
	defer f.Close()
	sum := sha256.New()
	if _, err := io.Copy(sum, f); err != nil {
		return copiedFile{}, err
	}
	return copiedFile{
		Path:   filepath.ToSlash(imagePath),
		SHA256: hex.EncodeToString(sum.Sum(nil)),
	}, nil
}
