// File: insecure_upload.go
// Demonstrates CWE-20: Improper Input Validation
//
// A tiny HTTP server that writes an uploaded file to disk using the
// user-supplied “filename” form-field without checking for “..”, absolute
// paths, or other dangerous characters.  An attacker can upload to
// “../../etc/crontab”, overwriting arbitrary files.

package main

import (
	"io"
	"log"
	"net/http"
	"os"
)

func main() {
	http.HandleFunc("/upload", uploadHandler)
	log.Println("Listening on :8080 …")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func uploadHandler(w http.ResponseWriter, r *http.Request) {
	// Parse multipart form (accept up to 10 MB file data)
	if err := r.ParseMultipartForm(10 << 20); err != nil {
		http.Error(w, "parse error", http.StatusBadRequest)
		return
	}

	// ────────────────────────────────────────────────────────────────
	// VULNERABLE CODE (CWE-20)
	// ────────────────────────────────────────────────────────────────
	// We trust the *exact* filename string provided by the attacker.
	filename := r.FormValue("filename") // e.g. "../../etc/passwd"
	fileData, _, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "file missing", http.StatusBadRequest)
		return
	}
	defer fileData.Close()

	dst, err := os.Create("/tmp/uploads/" + filename) // ← path traversal!
	if err != nil {
		http.Error(w, "save error", http.StatusInternalServerError)
		return
	}
	defer dst.Close()

	if _, err := io.Copy(dst, fileData); err != nil {
		http.Error(w, "write error", http.StatusInternalServerError)
		return
	}
	// ────────────────────────────────────────────────────────────────

	w.Write([]byte("file saved as " + filename + "\n"))
}

/*
Fix ideas:

1) Canonicalise and validate the filename:
      clean := filepath.Base(filename)                  // strips path
      ok, _ := regexp.MatchString(`^[a-zA-Z0-9_.-]+$`, clean)
      if !ok { …reject… }

2) Enforce a hard-coded upload directory with `filepath.Join` plus a
   post-`filepath.Clean` check that the result still resides under
   `/tmp/uploads`.

Both ensure untrusted input cannot escape the intended directory.
*/
