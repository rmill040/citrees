# Infrastructure Pointer

If you need to launch or monitor distributed jobs:

- use `citrees-exp --help`
- use `citrees-exp infra --help`
- inspect `paper/scripts/infra/`
- inspect `paper/scripts/api/`

Typical flow:

```bash
citrees-exp infra setup
citrees-exp infra s3
citrees-exp infra upload-data
citrees-exp infra launch-api
citrees-exp infra launch-workers --count 5
citrees-exp run
citrees-exp check
citrees-exp watch
citrees-exp infra terminate-workers
citrees-exp infra terminate-api
```
